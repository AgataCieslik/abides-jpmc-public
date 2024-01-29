from .trading_agent import TradingAgent
from .value_agent import ValueAgent
from typing import Optional, List, Dict
from abides_core import NanosecondTime, Message
import numpy as np
from math import ceil, sqrt, floor
from ..messages.recommendations import QuerySideRecommendation
from ..messages.query import QuerySpreadResponseMsg
from ..messages.trading_signal import BuySignal, SellSignal
from ..orders import Side
import queue
import logging

logger = logging.getLogger(__name__)


# TODO: określić wszedzie typy w wolnej chwili
# zakładam, że insider/informed trader handluje na tylko jednym rynku (ma uprzywilejowaną wiedzę na temat jednej spółki)
# zakładam że insider przez cały czas handluje w jedną stronę
class InformedAgent(ValueAgent):
    def __init__(self, id: int,
                 name: Optional[str] = None,
                 type: Optional[str] = None,
                 random_state: Optional[np.random.RandomState] = None,
                 symbol: str = "ABM",
                 starting_cash: int = 10_000_000,
                 sigma_n: float = 1000,
                 r_bar: int = 100_000,
                 kappa: float = 1.67e-15,
                 sigma_s: float = 100_000,
                 order_size_model=None,
                 # ten parametr chyba mógłby być opcjonalny
                 # albo mieć co najmniej inną wartość domyślną
                 lambda_a: float = 5.7e-12,
                 log_orders: float = True,
                 # nowe parametry
                 # rozmiar zlecenia
                 order_size: Optional[int] = None,
                 # TODO: upewnić się, że to właściwy typ
                 time_horizon: Optional[NanosecondTime] = None,
                 # minimalna ilość jaką chcemy nabyc/sprzedać
                 min_holdings: Optional[Dict[str, int]] = None,
                 # dać sensowniejszą wartośc domyślną na sigma_pv
                 # oczekiwany wzrost/spadek (wartość bezwzględna)
                 mean_pv: float = 0,
                 # odchylenie rozkładu
                 sigma_pv: float = 0,
                 # ograniczenie na ilość zapytań, na które odpowiadamy
                 queries_limit: Optional[int] = None,
                 # czy wysyłamy rekomendacje przed zrealizowaniem własnego planu minimum
                 signal_before_realizing: bool = False,
                 min_order_profit: float = 0,
                 # minimalna różnica ceny przy której decyduje się dać rekomendację
                 min_price_diff: Optional[float] = 0,
                 max_qty: Optional[int] = None
                 ):
        super().__init__(id, name, type, random_state, symbol, starting_cash, sigma_n, r_bar, kappa, sigma_s,
                         order_size_model, lambda_a, log_orders)
        # mamy dwie opcje:
        ## 1: podany model probabilistyczny, z którego losujemy rozmiar zlecenia
        ## 2: ustalony rozmiar zlecenia,jeśli nie jest podany w arg, jest losowy (odziedziczone po ValueAgent)
        if order_size_model is None:
            if order_size is not None:
                self.size = order_size

        # jeśli nie podaliśmy minimalnej ilości do kupienia/sprzedania, jest równa 0
        if min_holdings is None:
            self.min_holdings = {self.symbol: 0}
        else:
            self.min_holdings = min_holdings

        if max_qty is None:
            self.max_units_count = ceil(self.starting_cash / self.r_bar)
        else:
            self.max_units_count = max_qty
        self.time_horizon = time_horizon

        # wartość prywatna określa oszacowanie insidera nt. przyszłego wzrostu/spadku ceny
        # wartość k-1 indeksu to oczekiwany zysk z kupna kolejnej k-tej jednostki
        # zakładamy, że mean_pv jest niezerowa
        if mean_pv > 0:
            self.private_value = [int(x) for x in sorted(
                np.round(
                    self.random_state.normal(loc=mean_pv, scale=sqrt(sigma_pv), size=self.max_units_count).tolist()),
                reverse=True)]
        if mean_pv < 0:
            self.private_value = [int(x) for x in sorted(
                np.round(
                    self.random_state.normal(loc=mean_pv, scale=sqrt(sigma_pv), size=self.max_units_count).tolist()),
                reverse=False)]
            self.holdings[self.symbol] = self.max_units_count

        # jeśli True, wysyłaj rekomendacje niezależnie od realizacji swoich celów
        self.signal_before_realizing = signal_before_realizing

        # opcjonalne ograniczenie na liczbę wysyłanych rekomendacji
        if queries_limit is None:
            self.queries_limit = np.inf
        else:
            self.queries_limit = queries_limit

        # liczba wysłanych wiadomości
        self.signals_sent = 0

        # oczekujące wiadomości
        self.pending_queries: queue.PriorityQueue[Message] = queue.PriorityQueue()

        self.min_order_profit = min_order_profit
        self.min_price_diff = min_price_diff

        self.last_pred = None

        # TODO: zastanowić się, czy nie zastąpić tego metodą get_next_wakeup

    def get_holdings(self, symbol):
        if (self.symbol in self.holdings.keys()):
            return self.holdings[symbol]
        else:
            return 0

    def get_wakeup_schedule(self) -> None:
        # jeśli nie podaliśmy konkretnie czasu wydarzenia, używamy czasu zamknięcia rynku
        if (self.time_horizon is None) and (self.mkt_close is not None):
            self.time_horizon = self.mkt_close
        else:
            return

        if self.random_state is None:
            self.random_state = np.random.RandomState()

    def set_next_wakeup(self, current_time: NanosecondTime) -> None:
        delta = self.random_state.exponential(scale=1.0 / self.lambda_a)
        next_scheduled_wakeup = current_time + delta
        if next_scheduled_wakeup > self.time_horizon:
            return
        self.set_wakeup(next_scheduled_wakeup)

    def get_recommendation(self):
        current_value_ind = max(0, ceil(self.get_holdings(self.symbol)) - 1)
        final_fundamental = self.updateEstimates()

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)
        current_midprice = (bid + ask) / 2

        # jeśli szacowana cena na koniec dnia jest większa od aktualnej ceny, zwracamy sygnał BUY
        # uwaga: przyjmujemy założenie, że wszystkie zapytania dochodzą przed wydarzeniem
        if self.current_time < self.time_horizon:
            self.last_pred = final_fundamental + self.private_value[current_value_ind]

        if (self.last_pred - current_midprice) > self.min_price_diff:
            return BuySignal(symbol=self.symbol)
        elif (current_midprice - self.last_pred) > self.min_price_diff:
            return SellSignal(symbol=self.symbol)
        return

    def send_recommendation(self, recipient_id):
        response = self.get_recommendation()
        if response is not None:
            self.send_message(recipient_id=recipient_id, message=response)
            self.signals_sent += 1

    def get_order_size(self, side: Side) -> int | float:
        if self.size is not None:
            size = self.size
        elif self.order_size_model is not None:
            size = self.order_size_model.sample(random_state=self.random_state)

        max_size = None
        if side == Side.BID:
            max_size = self.max_buy_order_size()
        elif side == Side.ASK:
            max_size = self.max_sell_order_size()
        if size <= max_size:
            return size
        else:
            return max_size

    def private_value_for_buy_order(self, order_qty: float) -> float:
        current_ind = int(ceil(self.get_holdings(self.symbol)))
        order_qty_adj = int(ceil(order_qty))
        stop_ind = current_ind + order_qty_adj

        mean_private_value = 0
        if current_ind == len(self.private_value):
            return mean_private_value
        if stop_ind >= len(self.private_value):
            mean_private_value = np.mean(self.private_value[current_ind:])
        else:
            mean_private_value = np.mean(self.private_value[current_ind:stop_ind])
        if np.isnan(mean_private_value):
            print("beee")
        return mean_private_value

    def private_value_for_sell_order(self, order_qty: float) -> float:
        current_ind = int(ceil(self.get_holdings(self.symbol)))
        order_qty_adj = int(ceil(order_qty))
        start_ind = current_ind - order_qty_adj

        mean_private_value = 0
        if current_ind == 0:
            return mean_private_value
        if start_ind < 0:
            mean_private_value = np.mean(self.private_value[:current_ind])
        else:
            mean_private_value = np.mean(self.private_value[start_ind:current_ind])
        return mean_private_value

    def max_buy_order_size(self) -> float:
        current_ind = int(ceil(self.get_holdings(self.symbol)))
        return self.max_units_count - current_ind

    def max_sell_order_size(self) -> float:
        current_ind = int(ceil(self.get_holdings(self.symbol)))
        return current_ind

    def send_pending_recommendations(self):
        while (self.pending_queries.empty() == False):
            delivery_time, sender_id = self.pending_queries.get()
            if self.signals_sent < self.queries_limit:
                self.send_recommendation(sender_id)
            else:
                self.pending_queries = queue.PriorityQueue()
                break

    def receive_message(
            self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        TradingAgent.receive_message(self, current_time, sender_id, message)
        if self.state == "AWAITING_SPREAD":
            if isinstance(message, QuerySpreadResponseMsg):
                ## w pierwszej kolejności wykonujemy swoje akcje
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed:
                    return

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                self.placeOrder()

                ## w drugiej kolejności wysyłamy rekomendacje
                if self.signals_sent < self.queries_limit:
                    # wysyłamy rekomendacje tylko jeśli nie przekroczyliśmy limitu
                    if self.signal_before_realizing == True:
                        # jeśli wysyłamy rekomendacje niezależnie od własnych interesów, wysyłamy wszystkie oczekujące rekomendacje
                        self.send_pending_recommendations()
                    # jeśli posiadamy w ogóle dany symbol i mam go minimum tyle, ile chcemy
                    else:
                        current_holdings = self.get_holdings(self.symbol)
                        if current_holdings >= self.min_holdings[self.symbol]:
                            self.send_pending_recommendations()

                self.state = "AWAITING_WAKEUP"

        # jeśli otrzyma rekomendację, dodaje ją do kolejki
        if isinstance(message, QuerySideRecommendation):
            self.pending_queries.put((current_time, sender_id))

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times and market_open wakeup call.
        TradingAgent.wakeup(self, current_time)

        self.state = "INACTIVE"

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True
                self.get_wakeup_schedule()

                # Time to start trading!
                logger.debug("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        self.set_next_wakeup(current_time)

        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        # TODO: czy to zachowanie jest dla tego agenta pożądane?
        self.cancel_all_orders()

        if isinstance(self, InformedAgent):
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
        else:
            self.state = "ACTIVE"

    def placeOrder(self):
        r_T = self.updateEstimates()

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        if bid and ask:
            buy_order_size = self.get_order_size(side=Side.BID)
            sell_order_size = self.get_order_size(side=Side.ASK)

            buy_order_pv = self.private_value_for_buy_order(buy_order_size)
            sell_order_pv = self.private_value_for_sell_order(sell_order_size)

            # docelowa wartość potencjalnie dokupionych jednostek
            buy_belief = r_T + buy_order_pv
            # docelowa wartość potencjalnie sprzedanych jednostek
            sell_belief = r_T + sell_order_pv

            # nasz potencjalny zysk z kupna to różnica między naszym przekonaniem a aktualną ceną kupna
            # (jeśli aktywo jest niedoszacowane, opłaca nam się dokupić)
            expected_buy_profit = buy_order_size * (buy_belief - bid)
            # nasz potencjalny zysk ze sprzedaży to różnica między aktualną ceną sprzedaży a naszym przekonaniem
            # (jeśli aktywo jest przeszacowane, opłaca nam się sprzedać
            expected_sell_profit = sell_order_size * (ask - sell_belief)

            expected_profit = 0
            if expected_buy_profit >= expected_sell_profit:
                side = Side.BID
                expected_profit = expected_buy_profit
                p = bid
                order_size = buy_order_size
            else:
                side = Side.ASK
                expected_profit = expected_sell_profit
                p = ask
                order_size = sell_order_size

            if (expected_profit >= self.min_order_profit) and (order_size > 0):
                self.place_limit_order(self.symbol, order_size, side, p)
