import logging
from math import sqrt
from typing import Any, Dict, List

import numpy as np
from abides_core import NanosecondTime

from .sparse_mean_reverting_oracle import SparseMeanRevertingOracle

logger = logging.getLogger(__name__)


class SpecialEventOracle(SparseMeanRevertingOracle):
    def __init__(
            self,
            mkt_open: NanosecondTime,
            mkt_close: NanosecondTime,
            # w tym słowniku symbols określamy paramety megashocku - jak często i jak intensywnie do megashocków dochodzi
            symbols: Dict[str, Dict[str, Any]],
            special_events: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        super().__init__(mkt_open, mkt_close, symbols)
        # założenie: wydarzenia są posortowane według czasu występowania
        self.special_events = special_events

    def advance_fundamental_value_series(
            self, current_time: NanosecondTime, symbol: str
    ) -> int:
        """This method advances the fundamental value series for a single stock symbol,
        using the OU process.  It may proceed in several steps due to our periodic
        application of "megashocks" to push the stock price around, simulating
        exogenous forces."""

        # Generation of the fundamental value series uses a separate random state object
        # per symbol, which is part of the dictionary we maintain for each symbol.
        # Agent observations using the oracle will use an agent's random state object.
        s = self.symbols[symbol]

        # This is the previous fundamental time and value.
        pt, pv = self.r[symbol]

        # If time hasn't changed since the last advance, just use the current value.
        if current_time <= pt:
            return pv

        # Otherwise, we have some work to do, advancing time and computing the fundamental.

        # We may not jump straight to the requested time, because we periodically apply
        # megashocks to push the series around (not always away from the mean) and we need
        # to compute OU at each of those times, so the aftereffects of the megashocks
        # properly affect the remaining OU interval.

        mst = self.megashocks[symbol][-1]["MegashockTime"]
        msv = self.megashocks[symbol][-1]["MegashockValue"]

        # wyznaczamy eventy, które mają miejsce w okresie od ostatniego updatu do teraz
        special_events = self.special_events[symbol]
        current_events = [event for event in iter(special_events) if
                          ((event['SpecialEventTime'] <= current_time) and (event['SpecialEventTime'] > pt))]
        self.special_events_lens.append(len(current_events))

        if mst < current_time:
            while mst < current_time:
                # case 0: któryś z eventów (lub kilka z nich) jest przed kolejnym wyznaczonym megashockiem
                # zakładam eventy posortowane według czasu
                events_before_next_shock = [event for event in iter(current_events) if event['SpecialEventTime'] <= mst]
                if len(events_before_next_shock) > 0:
                    self.special_events_cond += 1
                    for event in events_before_next_shock:
                        event_time = event['SpecialEventTime']
                        event_value = event['SpecialEventValue']
                        v = self.compute_fundamental_at_timestamp(event_time, event_value, symbol, pt, pv)
                        pt, pv = event_time, v

                # A megashock is scheduled to occur before the new time to which we are advancing.  Handle it.

                # Advance time from the previous time to the time of the megashock using the OU process and
                # then applying the next megashock value.
                v = self.compute_fundamental_at_timestamp(mst, msv, symbol, pt, pv)

                # Update our "previous" values for the next computation.
                pt, pv = mst, v

                # Since we just surpassed the last megashock time, compute the next one, which we might or
                # might not immediately consume.  This works just like the first time (in __init__()).

                mst = pt + int(np.random.exponential(scale=1.0 / s["megashock_lambda_a"]))
                msv = s["random_state"].normal(
                    loc=s["megashock_mean"], scale=sqrt(s["megashock_var"])
                )
                msv = msv if s["random_state"].randint(2) == 0 else -msv

                self.megashocks[symbol].append(
                    {"MegashockTime": mst, "MegashockValue": msv}
                )

                # The loop will continue until there are no more megashocks before the time requested
                # by the calling method.
        elif len(current_events) > 0:
            for event in current_events:
                event_time = event['SpecialEventTime']
                event_value = event['SpecialEventValue']
                v = self.compute_fundamental_at_timestamp(event_time, event_value, symbol, pt, pv)
                pt, pv = event_time, v

        # Once there are no more megashocks to apply (i.e. the next megashock is in the future, after
        # current_time), then finally advance using the OU process to the requested time.
        v = self.compute_fundamental_at_timestamp(current_time, 0, symbol, pt, pv)

        return v
