import logging
from typing import Optional, Dict, Tuple, List

import numpy as np
from abides_core import Message, NanosecondTime

from ..trading_agent import TradingAgent
from ..value_agent import ValueAgent
from ...messages.recommendations import QueryFinalValue, FinalValueResponse, QuerySideRecommendation
from ...messages.trading_signal import BuySignal, SellSignal, TradingSignal
from ...orders import Side

logger = logging.getLogger(__name__)


# wakeup ---> fundamental value query DONE
# on receive fundamental value response ----> update estimates, zapisywać R_T -----> get current spread
# get current spread ----> place order ---> update estimates

# co powinno robić update esitmates?
# bierze ostatnią

# potrzebujemy tablicę na rekomendacje i metody
# metoda .query_insiders- jest
# @property received_all_news - > True jeśli mamy komplet rekomendacji od wszystkich agentów
# kiedy otrzymaliśmy komplet wieści --> selekcja --> update estimates --> get_current_spread --> dalej prawie tak samo jak zwykły value agent
class FollowerValueAgent(ValueAgent):
    def __init__(
            self,
            id: int,
            name: Optional[str] = None,
            type: Optional[str] = None,
            random_state: Optional[np.random.RandomState] = None,
            symbol: str = "ABM",
            starting_cash: int = 100_000,
            sigma_n: float = 10_000,
            r_bar: int = 100_000,
            kappa: float = 0.05,
            sigma_s: float = 100_000,
            order_size_model=None,
            lambda_a: float = 0.005,
            log_orders: float = False,
            contacts: Optional[List[int]] = None,
            delays: Optional[List[int]] = None
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state, symbol, starting_cash, sigma_n, r_bar, kappa, sigma_s,
                         order_size_model, lambda_a, log_orders, contacts, delays)
        self.prev_obs_time: Optional[NanosecondTime] = None
        self.last_news: Dict[int, FinalValueResponse] = {}
        self.final_fundamental: Optional[float] = None

    def reset_properties(self) -> None:
        super().reset_properties()
        self.prev_obs_time = None
        self.last_news = {}
        self.final_fundamental = None

    def query_insiders(self) -> None:
        message = QueryFinalValue(self.symbol)
        self.broadcast(message)

    @property
    def received_all_responses(self) -> bool:
        senders = list(self.last_news.keys())
        if sorted(senders) == sorted(self.contacts):
            return True
        return False

    def get_most_recent_estimate(self) -> Tuple[NanosecondTime, float]:
        news = [message for message in iter(self.last_news.values()) if message.r_T and message.obs_time]
        if len(news) > 0:
            sorted_messages = sorted(news, key=lambda message: message.obs_time, reverse=True)
            most_recent_message = sorted_messages[0]
            obs_time = most_recent_message.obs_time
            estimate = most_recent_message.r_T
            return obs_time, estimate
        else:
            return None, None

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times and market_open wakeup call.

        # chcemy wywołać metodę wakeup TradingAgenta a nie value agenta
        TradingAgent.wakeup(self, current_time)

        self.state = "INACTIVE"

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True

                # Time to start trading!
                logger.debug("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        self.set_wakeup(current_time + int(round(delta_time)))

        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.query_insiders(self.symbol)
            self.state = "AWAITING_NEWS"
            return

        self.cancel_all_orders()

        if isinstance(self, FollowerValueAgent):
            self.query_insiders()
            self.state = "AWAITING_NEWS"
        else:
            self.state = "ACTIVE"

    def placeOrder(self) -> None:
        r_T = self.final_fundamental

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        if bid and ask:
            mid = int((ask + bid) / 2)
            spread = abs(ask - bid)

            if self.random_state.rand() < self.percent_aggr:
                adjust_int = 0
            else:
                adjust_int = self.random_state.randint(
                    0, min(9223372036854775807 - 1, self.depth_spread * spread)
                )
                # adjustment to the limit price, allowed to post inside the spread
                # or deeper in the book as a passive order to maximize surplus

            if r_T < mid:
                # fundamental belief that price will go down, place a sell order
                buy = False
                p = (
                        bid + adjust_int
                )  # submit a market order to sell, limit order inside the spread or deeper in the book
            elif r_T >= mid:
                # fundamental belief that price will go up, buy order
                buy = True
                p = (
                        ask - adjust_int
                )  # submit a market order to buy, a limit order inside the spread or deeper in the book
        else:
            # initialize randomly
            buy = self.random_state.randint(0, 1 + 1)
            p = r_T

        # Place the order
        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)

        side = Side.BID if buy == 1 else Side.ASK
        self.side = side

        if self.size > 0:
            self.place_limit_order(self.symbol, self.size, side, p)

    def receive_message(
            self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)
        if self.state == "AWAITING_NEWS":
            if isinstance(message, FinalValueResponse):
                self.last_news[sender_id] = message
                if (self.received_all_responses == True):
                    obs_time, estimate = self.get_most_recent_estimate()
                    if estimate and obs_time:
                        self.prev_obs_time = obs_time
                        self.final_fundamental = estimate
                        self.last_news = {}
                        self.get_current_spread(self.symbol)
                        self.state = "AWAITING_SPREAD"

        if isinstance(message, QueryFinalValue):
            # when some FollowerValueAgent asks for fundamental value
            response = FinalValueResponse(symbol=self.symbol, obs_time=self.prev_obs_time, r_T=self.final_fundamental,
                                          sigma_t=self.sigma_t)

            delay = self.get_agent_delay(sender_id)
            self.send_message(recipient_id=sender_id, message=response, delay=delay)

        if isinstance(message, QuerySideRecommendation):
            delay = self.get_agent_delay(sender_id)
            if self.side is not None:
                if self.side == Side.BID:
                    response = BuySignal(symbol=self.symbol)
                elif self.side == Side.ASK:
                    response = SellSignal(symbol=self.symbol)
                self.send_message(recipient_id=sender_id, message=response, delay=delay)
            else:
                self.send_message(recipient_id=sender_id, message=TradingSignal(self.symbol), delay=delay)
