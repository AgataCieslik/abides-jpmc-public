from .trading_agent import TradingAgent
from .value_agent import ValueAgent
from typing import Optional, List, Dict
from abides_core import NanosecondTime, Message
import numpy as np
from math import ceil, sqrt, floor
from ..messages.recommendations import QuerySideRecommendation
from ..messages.trading_signal import BuySignal, SellSignal
from ..orders import Side
import queue
import logging

logger = logging.getLogger(__name__)


# TODO: określić wszedzie typy w wolnej chwili
# zakładam, że insider/informed trader handluje na tylko jednym rynku (ma uprzywilejowaną wiedzę na temat jednej spółki)
class InformedAgent(ValueAgent):
    def __init__(self, id: int,
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
                 # ten parametr chyba mógłby być opcjonalny
                 # albo mieć co najmniej inną wartość domyślną
                 lambda_a: float = 0.005,
                 log_orders: float = False,
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
                 # opcjonalna "sztywna" częstotliwość akcji
                 wakeup_freq: Optional[NanosecondTime] = None,
                 first_wake: Optional[NanosecondTime] = None,
                 wakeup_schedule: Optional[List[NanosecondTime]] = None,
                 min_order_profit: float = 0,
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

        # TODO: zastanowić się, czy używanie r_bar jest tutaj legalne, czy lepiej posłużyć się q_max?
        max_units_count = ceil(self.starting_cash / self.r_bar)
        self.time_horizon = time_horizon

        # wartość prywatna określa oszacowanie insidera nt. przyszłego wzrostu/spadku ceny
        # wartość k-1 indeksu to oczekiwany zysk z kupna kolejnej k-tej jednostki
        self.private_value = [int(x) for x in sorted(
            np.round(self.random_state.normal(loc=mean_pv, scale=sqrt(sigma_pv), size=max_units_count).tolist(),
                     reverse=True))]

        # jeśli True, wysyłaj rekomendacje niezależnie od realizacji swoich celów
        self.signal_before_realizing = signal_before_realizing

        # opcjonalne ograniczenie na liczbę wysyłanych rekomendacji
        if self.queries_limit is None:
            self.queries_limit = np.inf
        else:
            self.queries_limit = queries_limit

        # liczba wysłanych wiadomości
        self.signals_sent = 0

        # oczekujące wiadomości
        self.pending_queries: queue.PriorityQueue[Message] = queue.PriorityQueue()

        # opcjonalne regularne "przebudzenia"
        self.wakeup_freq = wakeup_freq

        # opcjonalny moment pierwszego przebudzenia
        self.first_wake = first_wake

        self.realized_wakeups: List[NanosecondTime] = []
        self.min_order_profit = min_order_profit

    # TODO: zastanowić się, czy nie zastąpić tego metodą get_next_wakeup
    def get_wakeup_schedule(self) -> None:
        # jesli mamy podane z gory czasy przebudzeń, nie modyfikujemy ich
        if self.scheduled_wakeups is not None:
            return

        # jeśli nie podaliśmy konkretnie czasu pierwszego przebudzenia, używamy czasu otwarcia rynku
        if self.first_wake is not None:
            first_wake = self.first_wake
        elif self.mkt_open is not None:
            first_wake = self.mkt_open
        else:
            return

        # jeśli nie podaliśmy konkretnie czasu wydarzenia, używamy czasu zamknięcia rynku
        if self.horizon is not None:
            last_wake = self.horizon
        elif self.mkt_close is not None:
            last_wake = self.mkt_close
        else:
            return

        scheduled_wakeups = queue.Queue()
        wake_time = first_wake

        if self.wake_freq is not None:
            while wake_time < last_wake:
                wake_time += self.wake_freq
                scheduled_wakeups.put(wake_time)

        elif self.lambda_a is not None:
            if self.random_state is None:
                random_state = np.random.RandomState()

            while wake_time < last_wake:
                delta_time = random_state.exponential(scale=1.0 / self.lambda_a)
                wake_time += int(round(delta_time))
                scheduled_wakeups.put(wake_time)
        scheduled_wakeups.put(last_wake)
        return scheduled_wakeups

    def set_next_wakeup(self, current_time: NanosecondTime):
        if self.scheduled_wakeups is not None:
            next_scheduled_wakeup = self.scheduled_wakeups.get()
            while (next_scheduled_wakeup <= current_time):
                next_scheduled_wakeup = self.scheduled_wakeups.get()
                if self.scheuled_wakeups.empty() == True:
                    return
            self.set_wakeup(next_scheduled_wakeup)
            self.realized_wakeups.append(next_scheduled_wakeup)

    def get_recommendation(self):
        current_value_ind = ceil(self.holdings[self.symbol]) - 1
        if self.private_value[current_value_ind] > 0:
            return BuySignal(symbol=self.symbol)
        return SellSignal(symbol=self.symbol)

    def send_recommendation(self, recipient_id):
        response = self.get_recommendation()
        self.send_message(recipient_id=recipient_id, message=response)
        self.signals_sent += 1

    def get_order_size(self) -> int | float:
        if self.size is not None:
            return self.size
        elif self.order_size_model is not None:
            return self.order_size_model.sample(random_state=self.random_state)

    def private_value_for_order(self, order_qty: float) -> float:
        current_ind = int(ceil(self.holdings[self.symbol]))
        order_qty_adj = int(ceil(order_qty))
        return np.mean(self.private_value[current_ind:(current_ind + order_qty_adj)])

    def send_pending_recommendations(self):
        while (self.pending_queries.empty() == False):
            delivery_time, sender_id = self.pending_queries.get()
            if self.signals_sent < self.queries_limit:
                self.send_recommendation(sender_id)
                self.signals_sent += 1
            else:
                self.pending_queries = queue.PriorityQueue()
                break

    def receive_message(
            self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(self, current_time, sender_id, message)
        if isinstance(message, QuerySideRecommendation):
            if self.signals_sent < self.queries_limit:
                # wysyłamy rekomendacje tylko jeśli nie przekroczyliśmy limitu
                if self.signal_before_realizing == True:
                    # jeśli wysyłamy rekomendacje niezależnie od własnych interesów
                    self.send_recommendation(sender_id)
                elif self.holdings[self.symbol] >= self.min_holdings[self.symbol]:
                    # jeśli wysyłamy rekomendacje dopiero po spełnieniu własnych oczekiwań
                    if self.pending_queries.empty() == False:
                        # jeśli są inne prośby oczekujące, wykonujemy je w pierwszej kolejności
                        self.pending_queries.put((current_time, sender_id))
                        self.send_pending_recommendations()
                    else:
                        self.send_recommendation(sender_id)

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times and market_open wakeup call.
        TradingAgent.wakeup(current_time)

        #TODO: czemu ten stan inactive? jaka jest logika za tym?
        self.state = "INACTIVE"

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True
                self.scheduled_wakeups = self.get_wakeup_schedule()

                # Time to start trading!
                logger.debug("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        self.set_next_wakeup()

        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        #TODO: czy to zachowanie jest dla tego agenta pożądane?
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
            mid = int((ask + bid) / 2)
            order_size = self.get_order_size()

            pv = self.private_value_for_order(order_size)
            belief = r_T + pv

            if mid < belief:
                # buy
                side = Side.BID
                p = bid
                expected_profit = order_size * (belief - bid)
            elif mid > belief:
                # sell
                side = Side.ASK
                p = ask
                expected_profit = order_size * (mid - ask)

            if expected_profit >= self.min_profit:
                self.place_limit_order(self.symbol, order_size, side, p)


