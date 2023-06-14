from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from abides_core.network import Network
from abides_core.utils import ns_date, fmt_ts

from .insider_value_agent import InsiderValueAgent


def insider_network_plot(network: Network):
    insiders_ids = [agent.id for agent in iter(network.agents) if isinstance(agent, InsiderValueAgent)]
    chosen_insider_id = insiders_ids[0]

    nx_graph = network.generate_networkx_object()
    shortests_paths_from_insider = dict(nx.single_source_shortest_path_length(nx_graph, chosen_insider_id))
    degrees = dict(nx_graph.degree)
    return nx.draw(nx_graph, with_labels=True, node_color=list(shortests_paths_from_insider.values()),
                   cmap=plt.cm.Reds_r, node_size=[v * 100 for v in degrees.values()])


def generate_bests_df(end_state: Dict, symbol: str, config_name: str) -> pd.DataFrame:
    order_book = end_state["agents"][0].order_books[symbol]
    L1 = order_book.get_L1_snapshots()

    best_bids = pd.DataFrame(L1["best_bids"], columns=["time", "price", "qty"])
    best_asks = pd.DataFrame(L1["best_asks"], columns=["time", "price", "qty"])

    best_bids.index = pd.to_datetime(best_bids['time'])
    best_asks.index = pd.to_datetime(best_asks['time'])

    best_bids = best_bids.drop(columns=['time'])
    best_asks = best_asks.drop(columns=['time'])

    best_bids = best_bids.resample("1s").last().ffill().bfill().reset_index()
    best_asks = best_asks.resample("1s").last().ffill().bfill().reset_index()

    best_bids = best_bids.rename(columns={"price": "best_bid_price", "qty": "best_bid_qty"})
    best_asks = best_asks.rename(columns={"price": "best_ask_price", "qty": "best_ask_qty"})

    bests = pd.merge(best_bids, best_asks, left_on=["time"], right_on=['time'], how="outer")

    bests["time"] = bests["time"].apply(lambda x: x.value - ns_date(x.value))
    bests['time'] = bests['time'].apply(lambda t: fmt_ts(t).split(" ")[1])
    bests['config_name'] = config_name

    return bests


def updated_holdings(logs: pd.DataFrame, config_name: str, symbol: str = "ABM") -> pd.DataFrame:
    filtered_logs = logs.loc[
        (logs['EventType'] == 'HOLDINGS_UPDATED'), ['EventTime', 'agent_id', 'agent_type', 'CASH', symbol]]
    filtered_logs['config_name'] = config_name
    return filtered_logs


def executed_orders(logs: pd.DataFrame, config_name: str) -> pd.DataFrame:
    filtered_logs = logs.loc[
        (logs['EventType'] == 'ORDER_EXECUTED'), ['EventTime', 'time_placed', 'EventType', 'agent_id', 'agent_type',
                                                  'side', 'quantity', 'fill_price', 'limit_price']]
    filtered_logs['config_name'] = config_name
    return filtered_logs


def final_surplus(logs: pd.DataFrame, config_name: str) -> pd.DataFrame:
    filtered_logs = logs.loc[
        logs['EventType'].isin(['FINAL_VALUATION', 'STARTING_CASH']), ['EventType', 'agent_id', 'agent_type',
                                                                       'ScalarEventValue']]
    filtered_logs = pd.pivot_table(filtered_logs, columns=['EventType'], index=['agent_id', 'agent_type'])
    filtered_logs['config_name'] = config_name
    return filtered_logs
