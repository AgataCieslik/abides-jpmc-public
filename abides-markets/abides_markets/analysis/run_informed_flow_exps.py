from abides_markets.configs import informed_flow_config
from abides_core import abides
from abides_markets.analysis.extract_data import *
from abides_core.utils import str_to_ns, parse_logs_df
import pickle
from datetime import datetime

# zaczynamy od ustawienia special eventu i parametrów wyroczni
date = "20240126"
event_time = int(pd.to_datetime(date).to_datetime64()) + str_to_ns("12:00:00")
event_value = 100000 * 0.015

event_params = {"SpecialEventTime": event_time, "SpecialEventValue": event_value}

max_qts = [10000]
order_sizes = [[25, 100, 1000]]
followers_num = [10, 50, 100, 500, 1000]
num_of_replications = 25
agg_freq = "1min"

for k in range(num_of_replications):
    seed = int(datetime.now().timestamp() * 1_000_000) % (2 ** 32 - 1)
    for f in range(5):
        for i in range(1):
            for j in range(3):
                try:
                    config = informed_flow_config.build_config(date=date,
                                                               seed=seed, special_event=event_params,
                                                               informed_agent_kwargs={"order_size": order_sizes[i][j],
                                                                                      "max_qty": max_qts[i]},
                                                               num_followers=followers_num[f])
                    end_state = abides.run(config)
                    prices = extract_spread(end_state, "1min")
                    volume = extract_transacted_volume(end_state, "1min")
                    logs_df = parse_logs_df(end_state)
                    informer_actions = extract_informer_actions(logs_df)
                    holdings = extract_holdings_updates(logs_df)
                    surplus = extract_surplus(logs_df)
                    data = {"prices": prices, "volume": volume, "informer_actions": informer_actions,
                            "holdings": holdings,
                            "surplus": surplus}
                    pickle.dump(data,
                                open(f"../../results/informed_flow/data_informed_flow_{max_qts[i]}_{order_sizes[i][j]}_{followers_num[f]}_rep_{k}.pkl", "wb"))
                    print(f"Suceeded: ({max_qts[i]}, {order_sizes[i][j]}, {followers_num[f]}), rep. {k}")
                except Exception as e:
                    print(f"Failed: ({max_qts[i]}, {order_sizes[i][j]}, {followers_num[f]}), rep. {k}; error: {e}")
