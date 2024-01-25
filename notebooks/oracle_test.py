#%%
from abides_markets.oracles.special_event_oracle import SpecialEventOracle
from abides_markets.oracles.sparse_mean_reverting_oracle import SparseMeanRevertingOracle
from abides_core.utils import str_to_ns
import numpy as np
#%%
oracle_params = {
    "ABM": {
        "r_bar": 100_000,
        "kappa": 1.67e-16,
        "sigma_s": 0,
        "fund_vol": 5e-5,
        "megashock_lambda_a": 2.77778e-18,
        "megashock_mean": 1000,
        "megashock_var": 50_000,
        #TODO: dtype: czy to jest mi niezbędne, dlaczego u innych może to działać? inna wersja bibliotek?
        "random_state": np.random.RandomState(
            seed=np.random.randint(low=0, high=2 ** 32, dtype=np.int64)
        ),
    }
}
#%%
special_events = {"ABM": [{"SpecialEventTime": 1 + str_to_ns("07:00:00"),"SpecialEventValue": 800}]}

#special_events = {"ABM": []}
#%% md

#%%
test_oracle = SpecialEventOracle(mkt_open=1, mkt_close=1 + str_to_ns("08:00:00"), symbols=oracle_params, special_events=special_events)
#%%
test_oracle_no_events = SparseMeanRevertingOracle(mkt_open=1, mkt_close=1 + str_to_ns("08:00:00"), symbols=oracle_params)
#%%
vals = [test_oracle.observe_price("ABM",t, np.random.RandomState(
                seed=np.random.randint(low=0, high=2**32, dtype=np.int64))) for t in np.arange(1,1 + str_to_ns("08:00:00"), 100000000000)]
#%%
vals2 = [test_oracle_no_events.observe_price("ABM",t, np.random.RandomState(
                seed=np.random.randint(low=0, high=2**32, dtype=np.int64))) for t in np.arange(1,1 + str_to_ns("08:00:00"), 100000000000)]
#%%
import plotly.express as px
#%%
px.line(x=np.arange(1,1 + str_to_ns("08:00:00"), 100000000000), y=vals)
#%%
px.line(x=np.arange(1,1 + str_to_ns("08:00:00"), 100000000000), y=vals2)
#%%
print(test_oracle.megashocks)
#%%
print(test_oracle_no_events.megashocks)