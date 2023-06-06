# startuje w stanie 'awaiting_signal'
# on receive -> jeśli dostaje trading signal przechodzi w stan 'awaiting_spread' - czekając na informację z giełdy o cenie
# dostaje informację z giełdy -> przechodzi w stan 'order_sent'
# nie przyjmuje więcej wiadomości - załóżmy setup że jedna symulacja = jedno wydarzenie na giełdzie
# poza tym follower ma zastąpić noise agent

# wariant B: reaguje na sygnał inny niż poprzedni
from ..trading_agent import TradingAgent
from typing import Optional, List
import numpy as np
from abides_core.message_processing_model import MessageProcessingModel, MostRecentProcessingModel
from abides_core import Message, NanosecondTime
from ...messages.trading_signal import TradingSignal, BuySignal, SellSignal
from ...messages.query import QuerySpreadResponseMsg
from ...generators import OrderSizeGenerator
from ...orders import Side
from abc import abstractmethod


# "zwykły" follower wysyła zlecenie tylko raz przez cały okres symulacji
class FollowerAgent(TradingAgent):
    def __init__(self, id: int, name: Optional[str] = None,
                 type: Optional[str] = None,
                 random_state: Optional[np.random.RandomState] = None,
                 starting_cash: int = 100000,
                 log_orders: bool = False,
                 contacts: Optional[List[int]] = None,
                 delays: Optional[List[int]] = None,
                 wake_up_freq: NanosecondTime = 1_000_000_000,
                 poisson_arrival: bool = True,
                 order_size_model: Optional[OrderSizeGenerator] = None,
                 order_size: Optional[int] = None,
                 symbol: str = "ABM") -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders, contacts, delays)
        self.wake_up_freq: str = wake_up_freq
        self.poisson_arrival: bool = (
            poisson_arrival
        )
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq
        self.order_size_model = order_size_model
        self.size = order_size
        self.state = "AWAITING_WAKEUP"
        self.symbol = symbol
        # może bardziej elegancko byłoby zaimplementować self.position (long zamiast bid, short zamiast ask
        self.side = None

    def wakeup(self, current_time: NanosecondTime) -> None:
        can_trade = super().wakeup(current_time)
        if can_trade:
            self.state = "AWAITING_NEWS"

    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))

    def place_order(self, side: Side, limit_price: float) -> None:
        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)

        if self.size > 0:
            # TODO: zastanowić się logicznie, czy to powinno być zlecenie limit czy market
            self.place_limit_order(symbol=self.symbol, quantity=self.size, side=side, limit_price=limit_price)
            #self.place_market_order(symbol=self.symbol, quantity=self.size, side=side)

    def receive_message(self, current_time: NanosecondTime, sender_id: int, message: Message) -> None:
        super().receive_message(current_time, sender_id, message)
        if isinstance(message, (BuySignal, SellSignal)):
            if (message.symbol == self.symbol) and (self.state == "AWAITING_NEWS"):
                self.get_current_spread(self.symbol, depth=1)
                if isinstance(message, BuySignal):
                    self.state = "RECEIVED_BUY_SIGNAL"
                else:
                    self.state = "RECEIVED_SELL_SIGNAL"

        if isinstance(message, QuerySpreadResponseMsg):
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
            if bid and ask:
                # upewnić się, czy nie pomyliłam bidów z askami
                if self.state == "RECEIVED_SELL_SIGNAL":
                    self.place_order(Side.ASK, ask)
                    self.side = Side.ASK
                elif self.state == "RECEIVED_BUY_SIGNAL":
                    self.place_order(Side.BID, bid)
                    self.side = Side.BID
                self.state = "TRADING"
                self.broadcast(message)
            else:
                # TODO: do rozważenia - tutaj zakładam, że odpytujemy do skutku, ale może alternatywą powinno być zlecenie market?
                self.get_current_spread(self.symbol, depth=1)


# regularny follower wysyła zlecenie za każdym razem, gdy otrzyma sygnał inny od poprzedniego
class RegularFollowerAgent(FollowerAgent):
    def receive_message(self, current_time: NanosecondTime, sender_id: int, message: Message) -> None:
        super().receive_message(current_time, sender_id, message)
        if isinstance(message, (BuySignal, SellSignal)):
            if (message.symbol == self.symbol) and (self.state == "TRADING"):
                if isinstance(message, BuySignal):
                    message_side = Side.BID
                else:
                    message_side = Side.ASK
                if (message_side != self.side):
                    # anulujemy niezrealizowane zlecenie
                    self.cancel_all_orders()
                    self.get_current_spread(self.symbol, depth=1)
