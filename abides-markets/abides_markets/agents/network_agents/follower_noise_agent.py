import logging
from typing import Optional, List, Dict

import numpy as np
from abides_core import Message, NanosecondTime
from abides_core.utils import get_wake_time, str_to_ns

from ..noise_agent import NoiseAgent
from ..trading_agent import TradingAgent
from ...generators import OrderSizeGenerator
from ...messages.query import QuerySpreadResponseMsg
from ...messages.recommendations import QuerySideRecommendation, QueryFinalValue, FinalValueResponse
from ...messages.trading_signal import TradingSignal, BuySignal, SellSignal
from ...orders import Side

logger = logging.getLogger(__name__)


# przy obudzeniu pyta się o sygnały i podobnie jak follower value agent czeka na wszystkie
# na podstawie głosowania lub ostatniego najświeższego sygnału podejmuje decyzję buy/sell i przekazuje ją placeOrder
class FollowerNoiseAgent(NoiseAgent):
    """
    Noise agent implement simple strategy. The agent wakes up once and places 1 order.
    """

    def __init__(
            self,
            id: int,
            name: Optional[str] = None,
            type: Optional[str] = None,
            random_state: Optional[np.random.RandomState] = None,
            symbol: str = "ABM",
            starting_cash: int = 100000,
            log_orders: bool = False,
            order_size_model: Optional[OrderSizeGenerator] = None,
            # to jest średni pomysł, by tu był None, powinna być jakaś wartość liczbowa (dotyczy też NoiseAgent)
            wakeup_time: Optional[NanosecondTime] = None,
            contacts: Optional[List[int]] = None,
            delays: Optional[List[int]] = None
    ) -> None:

        # Base class init.
        super().__init__(id, name, type, random_state, symbol, starting_cash, log_orders, order_size_model, wakeup_time,
                         contacts, delays)
        self.last_recommendations: Dict[int, TradingSignal] = {}
        self.side = None

        # atrybuty robocze
        self.final_recoms = []
        self.messages = []
        self.wakeups = 0
        self.messages_sent = []

    def get_recommendations(self) -> None:
        message = QuerySideRecommendation(self.symbol)
        self.messages_sent.append(message)
        self.broadcast(message)

    @property
    def received_all_responses(self) -> bool:
        senders = list(self.last_recommendations.keys())
        if sorted(senders) == sorted(self.contacts):
            return True
        return False

    def final_recommendation(self) -> TradingSignal:
        # zakładam głosowanie większościowe
        message_occurencies = {}
        message_passed = None

        for sender, signal in self.last_recommendations.items():
            message_type = type(signal)
            if message_type == 'TradingSignal':
                continue
            if message_type not in message_occurencies.keys():
                message_occurencies[message_type] = 1
            else:
                message_occurencies[message_type] = message_occurencies[message_type] + 1
            if max(message_occurencies, key=message_occurencies.get) == message_type:
                message_passed = signal
        if isinstance(message_passed, BuySignal):
            self.final_recoms.append(message_passed)
            return Side.BID
        elif isinstance(message_passed, SellSignal):
            self.final_recoms.append(message_passed)
            return Side.ASK
        elif message_passed is None:
            self.final_recoms.append(None)
            return None

    def wakeup(self, current_time: NanosecondTime) -> None:
        self.wakeups += 1
        # Parent class handles discovery of exchange times and market_open wakeup call.
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

        if self.wakeup_time > current_time:
            self.set_wakeup(self.wakeup_time)
            return

        if self.mkt_closed and self.symbol not in self.daily_close_price:
            self.get_current_spread()
            self.state = "AWAITING_SPREAD"
            return

        if type(self) == FollowerNoiseAgent:
            self.get_recommendations()
            self.state = "AWAITING_ADVICE"
        else:
            self.state = "ACTIVE"

    def placeOrder(self) -> None:
        # place order in random direction at a mid
        if self.side.is_bid():
            buy_indicator = 1
        elif self.side.is_ask():
            buy_indicator = 0

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)

        if self.size > 0:
            if buy_indicator == 1 and ask:
                self.place_limit_order(self.symbol, self.size, Side.BID, ask)
            elif not buy_indicator and bid:
                self.place_limit_order(self.symbol, self.size, Side.ASK, bid)

    def receive_message(
            self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receive_message(current_time, sender_id, message)
        self.messages.append((current_time, sender_id, message))
        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == "AWAITING_ADVICE":
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if isinstance(message, TradingSignal):
                self.last_recommendations[sender_id] = message
                if self.received_all_responses == True:
                    recommendation = self.final_recommendation()
                    self.side = recommendation
                    if recommendation:
                        #TODO: do rozważenia
                        #self.logEvent("RECEIVED_SIDE_RECOMMENDATION", recommendation)
                        self.last_recommendations = {}
                        self.get_current_spread(self.symbol)
                        self.state = "AWAITING_SPREAD"
                    # TODO: czy to jest logicznie dobry rozkład? czy może powinien być lewoskośny? albo z innymi parametrami?
                    else:
                        self.set_wakeup(get_wake_time(current_time + str_to_ns('5min'), self.mkt_close))
                        self.state = "AWAITING_WAKEUP"

        if self.state == "AWAITING_SPREAD":
            if isinstance(message, QuerySpreadResponseMsg):

                if self.mkt_closed:
                    return

                self.placeOrder(side=self.side)
                self.state = "AWAITING_WAKEUP"

        if isinstance(message, QueryFinalValue):
            response = FinalValueResponse(symbol=self.symbol, obs_time=None, r_T=None,
                                          sigma_t=None)

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
