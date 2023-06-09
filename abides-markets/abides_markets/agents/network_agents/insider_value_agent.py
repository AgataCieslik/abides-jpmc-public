import logging
from typing import Optional, List

import numpy as np

from abides_core import Message, NanosecondTime

from ...messages.query import QuerySpreadResponseMsg
from ...messages.recommendations import QueryFinalValue, FinalValueResponse, QuerySideRecommendation
from ...messages.trading_signal import TradingSignal, BuySignal, SellSignal
from ...orders import Side
from abides_core import NanosecondTime
from ..value_agent import ValueAgent

logger = logging.getLogger(__name__)


class InsiderValueAgent(ValueAgent):
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
            delays: Optional[List[int]] = None,
            horizon: NanosecondTime = 0
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state, symbol, starting_cash, sigma_n, r_bar, kappa, sigma_s,
                         order_size_model, lambda_a, log_orders, contacts, delays)

        # The agent must track its previous observation time, so it knows how many time
        # units have passed.
        self.prev_obs_time: Optional[NanosecondTime] = None
        self.horizon = horizon
        self.final_fundamental: Optional[float] = None

    # kernel_starting, kernel_stopping, wakeup - chyba nie wymagają modyfikacji
    def updateEstimates(self) -> int:
        # Called by a background agent that wishes to obtain a new fundamental observation,
        # update its internal estimation parameters, and compute a new total valuation for the
        # action it is considering.

        # The agent obtains a new noisy observation of the current fundamental value
        # and uses this to update its internal estimates in a Bayesian manner.

        # potencjalnie: horizon = self.horizon_model.get()
        current_obs_time = self.current_time + self.horizon

        # pobieramy informację "z przyszłości"
        obs_t = self.oracle.observe_price(
            self.symbol,
            current_obs_time,
            sigma_n=self.sigma_n,
            random_state=self.random_state,
        )

        logger.debug("{} observed {} at {}", self.name, obs_t, self.current_time)

        # Update internal estimates of the current fundamental value and our error of same.

        # If this is our first estimate, treat the previous observation time as "market open".
        if self.prev_obs_time is None:
            self.prev_obs_time = self.mkt_open

        # First, obtain an intermediate estimate of the fundamental value by advancing
        # time from the previous wake time to the current time, performing mean
        # reversion at each time step.

        # delta must be integer time steps since last observation
        delta = current_obs_time - self.prev_obs_time

        # Update r estimate for time advancement.
        r_tprime = (1 - (1 - self.kappa) ** delta) * self.r_bar
        r_tprime += ((1 - self.kappa) ** delta) * self.r_t

        # Update sigma estimate for time advancement.
        sigma_tprime = ((1 - self.kappa) ** (2 * delta)) * self.sigma_t
        sigma_tprime += (
                                (1 - (1 - self.kappa) ** (2 * delta)) / (1 - (1 - self.kappa) ** 2)
                        ) * self.sigma_s

        # Apply the new observation, with "confidence" in the observation inversely proportional
        # to the observation noise, and "confidence" in the previous estimate inversely proportional
        # to the shock variance.
        self.r_t = (self.sigma_n / (self.sigma_n + sigma_tprime)) * r_tprime
        self.r_t += (sigma_tprime / (self.sigma_n + sigma_tprime)) * obs_t

        self.sigma_t = (self.sigma_n * self.sigma_t) / (self.sigma_n + self.sigma_t)

        # Now having a best estimate of the fundamental at time t, we can make our best estimate
        # of the final fundamental (for time T) as of current time t.  Delta is now the number
        # of time steps remaining until the simulated exchange closes.
        delta = max(0, (self.mkt_close - current_obs_time))

        # zastanowić się nad poniższą ideą - czy to wprowadzać?
        # IDEA: instead of letting agent "imagine time forward" to the end of the day,
        #       impose a maximum forward delta, like ten minutes or so.  This could make
        #       them think more like traders and less like long-term investors.  Add
        #       this line of code (keeping the max() line above) to try it.
        # delta = min(delta, 1000000000 * 60 * 10)

        r_T = (1 - (1 - self.kappa) ** delta) * self.r_bar
        r_T += ((1 - self.kappa) ** delta) * self.r_t

        # Our final fundamental estimate should be quantized to whole units of value.
        r_T = int(round(r_T))

        # Finally (for the final fundamental estimation section) remember the current
        # time as the previous wake time.
        self.prev_obs_time = current_obs_time

        logger.debug(
            "{} estimates r_T = {} as of {}", self.name, r_T, self.current_time
        )

        self.final_fundamental = r_T
        return r_T

    # placeOrder - chyba nie ma konieczności zmian

    def receive_message(
            self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receive_message(current_time, sender_id, message)

        # tutaj potrzebujemy dopisać:
        # reakcję na zapytanie o rekomendację
        # reakcję na zapytanie o wartość fundamentalną

        if isinstance(message, QueryFinalValue):
            # when some FollowerValueAgent asks for fundamental value
            response = FinalValueResponse(symbol=self.symbol, obs_time=self.prev_obs_time, r_T=self.final_fundamental,
                                                sigma_t=self.sigma_t)

            delay = self.get_agent_delay(sender_id)
            self.send_message(recipient_id=sender_id, message=response, delay=delay)

        if isinstance(message, QuerySideRecommendation):
            if self.side is not None:
                if self.side == Side.BID:
                    response = BuySignal(symbol=self.symbol)
                elif self.side == Side.ASK:
                    response = SellSignal(symbol=self.symbol)
                delay = self.get_agent_delay(sender_id)
                self.send_message(recipient_id=sender_id, message=response, delay=delay)

# get wake frequency - nie ma potrzeby modyfikowania

