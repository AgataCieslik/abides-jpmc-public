from abides_markets.agents.trading_agent import TradingAgent
from typing import Optional, List
import numpy as np
from abides_markets.generators import OrderSizeGenerator
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_core import Message, NanosecondTime
from abides_markets.messages.trading_signal import BuySignal, SellSignal
from abides_markets.orders import Side


# czy zagwarantować, że wyśle tylko jedno zlecenie? - nie trzeba
# jak zagwarantować, że obudzi się we właściwym momencie?
# może tak: rozszerzamy wyrocznię o metodę zwracającą informację o sygnale 'buy/hold/sell'
# dla ścisłości - insider dostaje cynk, że cena pójdzie w góre lub w dół, ale nie wie dokładnie o ile
# insider odpytuje wyrocznię i dostaje informację o sygnale
# w momencie gdy otrzyma sygnał buy lub sell postępuje zgodnie z nim i przesyła dalej

class InsiderAgent(TradingAgent):
    def __init__(self,
                 id: int,
                 name: Optional[str] = None,
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
                 signal_pct_change_threshold: int = 0,
                 horizon: NanosecondTime = 1_000_000_000,
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
        self.symbol = symbol
        self.pct_change_threshold = signal_pct_change_threshold
        self.horizon = horizon

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        self.oracle = self.kernel.oracle

    def wakeup(self, current_time: NanosecondTime) -> None:
        can_trade = super().wakeup(current_time)
        if can_trade:
            self.get_current_spread(self.symbol)

    # to w sumie mogłoby być zaimplementowane w trading agencie - przeoczenie twórców?
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

    def receive_message(self, current_time: NanosecondTime, sender_id: int, message: Message) -> None:
        super().receive_message(current_time, sender_id, message)
        if isinstance(message, QuerySpreadResponseMsg):

            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
            if bid and ask:
                current_price = (bid + ask) / 2

                # czy insider powinien dostawać zaszumioną informację ?
                fundamental_value = self.oracle.observe_price(self.symbol, self.current_time + self.horizon, sigma_n=0,
                                                              noisy=False)
                price_difference = fundamental_value - current_price
                abs_pct_change = abs(price_difference) / current_price

                # upewnić się, czy nie pomyliłam bidów z askami
                if abs_pct_change > self.pct_change_threshold:

                    # spodziewamy się wzrostu ceny, wysyłamy sygnał 'BUY'
                    if price_difference > 0:
                        message = BuySignal(self.symbol)
                        side = Side.BID
                        price = bid

                    # spodziewamy się spadku ceny, wysyłamy sygnał 'SELL'
                    elif price_difference < 0:
                        message = SellSignal(self.symbol)
                        side = Side.ASK
                        price = ask
                    self.place_order(side, price)
                    self.broadcast(message)

        self.set_wakeup(current_time + self.get_wake_frequency())
