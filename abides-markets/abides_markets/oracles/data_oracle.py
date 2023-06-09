from .oracle import Oracle
from abides_core import Message, NanosecondTime
from typing import Dict, Any, List, Optional
import pandas as pd
import datetime as dt
import logging
from math import sqrt
import numpy as np

logger = logging.getLogger(__name__)


class DataOracle(Oracle):
    def __init__(
            self,
            mkt_open: NanosecondTime,
            mkt_close: NanosecondTime,
            data: Dict[str, pd.Series],
            symbols: List[str],
            lag: int = 0
    ) -> None:
        # Symbols must be a dictionary of dictionaries with outer keys as symbol names and
        # inner keys: r_bar, kappa, sigma_s.
        self.mkt_open: NanosecondTime = mkt_open
        self.mkt_close: NanosecondTime = mkt_close
        self.symbols: List[str] = symbols
        self.data: Dict[str, pd.Series] = data
        self.lag: int = lag

        # The dictionary r holds the fundamental value series for each symbol.
        self.r: Dict[str, pd.Series] = {}

        then = dt.datetime.now()

        for symbol in symbols:
            logger.debug(
                "DataOracle computing fundamental value series for {}", symbol
            )
            self.r[symbol] = self.generate_fundamental_value_series(symbol=symbol)

        now = dt.datetime.now()

        logger.debug("DataOracle initialized for symbols {}", symbols)
        logger.debug("DataOracle initialization took {}", now - then)

    def generate_fundamental_values_series(self, symbol):
        time_series = self.data[symbol].copy()
        time_series = time_series.asfreq('N').ffill()

        series_start = time_series.index.min() + pd.to_timedelta(self.lag, unit='ns')
        simulation_start = dt.datetime(self.mkt_open)
        time_difference = simulation_start - series_start

        time_series.index = time_series.index + time_difference + + pd.to_timedelta(self.lag, unit='ns')

        time_series = (time_series.loc[time_series.index < self.mkt_close, :].copy()) * 100
        return int(time_series.round())

    # może warto byłoby poniższe metody wbić w jakąś nadklasę? może dopisać do oracle?
    def get_daily_open_price(
            self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int:
        """Return the daily open price for the symbol given.

        In the case of the MeanRevertingOracle, this will simply be the first
        fundamental value, which is also the fundamental mean. We will use the
        mkt_open time as given, however, even if it disagrees with this.
        """

        # If we did not already know mkt_open, we should remember it.
        if (mkt_open is not None) and (self.mkt_open is None):
            self.mkt_open = mkt_open

        logger.debug(
            "Oracle: client requested {symbol} at market open: {}", self.mkt_open
        )

        open_price = self.r[symbol].loc[self.mkt_open]
        logger.debug("Oracle: market open price was was {}", open_price)

        return open_price

    def observe_price(
            self,
            symbol: str,
            current_time: NanosecondTime,
            random_state: Optional[np.random.RandomState] = None,
            sigma_n: int = 1000
    ) -> int:
        if current_time >= self.mkt_close:
            r_t = self.r[symbol].loc[self.mkt_close - 1]
        else:
            r_t = self.r[symbol].loc[current_time]

            # Generate a noisy observation of fundamental value at the current time.
        if sigma_n == 0:
            obs = r_t
        else:
            obs = int(round(random_state.normal(loc=r_t, scale=sqrt(sigma_n))))

        logger.debug("Oracle: current fundamental value is {} at {}", r_t, current_time)
        logger.debug("Oracle: giving client value observation {}", obs)

        return obs

