from .trading_agent import TradingAgent
from .noise_agent import NoiseAgent
from abides_core import NanosecondTime
from typing import Optional, List
import numpy as np
import logging
from abides_core.message import Message
from ..messages.recommendations import QuerySideRecommendation
from ..messages.trading_signal import TradingSignal, BuySignal, SellSignal
from ..messages.query import QuerySpreadResponseMsg
from ..generators import OrderSizeGenerator
from ..orders import Side

logger = logging.getLogger(__name__)


class FollowerAgent(NoiseAgent):
    def __init__(
            self,
            id: int,
            name: Optional[str] = None,
            type: Optional[str] = None,
            random_state: Optional[np.random.RandomState] = None,
            symbol: str = "ABM",
            starting_cash: int = 100000,
            log_orders: bool = True,
            order_size_model: Optional[OrderSizeGenerator] = None,
            wakeup_time: Optional[NanosecondTime] = None,
            contacts: Optional[List[int]] = None,
            delays: Optional[List[int]] = None,
            time_horizon: NanosecondTime = None,
            informed_agent_id: int = None,
            order_size: Optional[int]=None
    ) -> None:
        super().__init__(id, name, type, random_state, symbol, starting_cash, log_orders, order_size_model,
                         wakeup_time, contacts, delays)
        self.time_horizon = time_horizon
        self.informed_agent_id = informed_agent_id
        self.requests_sent = 0
        if order_size is not None:
            self.size = order_size
        self.side = None

    def request_recommendation(self) -> None:
        message = QuerySideRecommendation(self.symbol)
        if self.informed_agent_id is not None:
            self.send_message(recipient_id=self.informed_agent_id, message=message)
            self.requests_sent += 1
        elif self.contacts is not None:
            self.broadcast(message=message)
            self.requests_sent += len(self.contacts)

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times and market_open wakeup call.
        TradingAgent.wakeup(self,current_time)

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
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        if current_time > self.time_horizon:
            return

        if type(self) == FollowerAgent:
            self.request_recommendation()
            self.state = "AWAITING_SIGNAL"
        else:
            self.state = "ACTIVE"

    def receive_message(
            self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # Parent class schedules market open wakeup call once market open/close times are known.
        TradingAgent.receive_message(self,current_time, sender_id, message)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.
        if self.state == "AWAITING_SIGNAL":
            # jeśli otrzymaliśmy rekomendację i do zdarzenia jeszcze nie doszło, wysyłamy zapytanie o aktualne ceny
            if (isinstance(message, TradingSignal)) and (current_time < self.time_horizon):

                if isinstance(message, BuySignal):
                    self.side = Side.BID

                elif isinstance(message, SellSignal):
                    self.side = Side.ASK

                self.get_current_spread(self.symbol)
                self.state = "AWAITING_SPREAD"

        if self.state == "AWAITING_SPREAD":
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if isinstance(message, QuerySpreadResponseMsg):
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed:
                    return

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                self.placeOrder()
                self.state = "AWAITING_WAKEUP"

    def placeOrder(self):
        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)

        if self.size > 0:
            if self.side.is_bid() and ask:
                self.place_limit_order(self.symbol, self.size, Side.BID, ask)
            elif self.side.is_ask() and bid:
                self.place_limit_order(self.symbol, self.size, Side.ASK, bid)
