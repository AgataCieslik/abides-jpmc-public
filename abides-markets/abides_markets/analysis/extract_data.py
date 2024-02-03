import pandas as pd
import numpy as np
from functools import partial


def extract_spread(end_state, freq):
    order_book = end_state["agents"][0].order_books["ABM"]
    L1 = order_book.get_L1_snapshots()
    best_bids = pd.DataFrame(L1["best_bids"], columns=["time", "price", "qty"])
    best_asks = pd.DataFrame(L1["best_asks"], columns=["time", "price", "qty"])

    for df in [best_bids, best_asks]:
        df['time'] = pd.to_datetime(df['time'])
        df.index = df['time']

    best_bids = best_bids.resample(freq).agg({"price": np.mean, 'qty': np.sum})
    best_asks = best_asks.resample(freq).agg({"price": np.mean, 'qty': np.sum})

    best_bids = best_bids.rename(columns={"price": "bid_price", "qty": "bid_qty"})
    best_asks = best_asks.rename(columns={"price": "ask_price", "qty": "ask_qty"})

    bests = best_bids.join(best_asks, on="time", how="outer")
    bests = bests.sort_values(by="time")
    bests['bid_price'] = bests['bid_price'].fillna(method="ffill")
    bests['ask_price'] = bests['ask_price'].fillna(method="ffill")
    bests['bid_qty'] = bests['bid_qty'].fillna(0)
    bests['ask_qty'] = bests['ask_qty'].fillna(0)
    bests["mid_price"] = (bests['ask_price'] + bests['bid_price']) / 2
    return bests


def extract_transacted_volume(end_state, freq):
    order_book = end_state["agents"][0].order_books["ABM"]
    buy_txs = pd.DataFrame.from_records(order_book.buy_transactions, columns=["time", "qty"])
    sell_txs = pd.DataFrame.from_records(order_book.sell_transactions, columns=["time", "qty"])

    for df in [buy_txs, sell_txs]:
        df['time'] = pd.to_datetime(df['time'])
        df.index = df['time']

    buy_txs = buy_txs.resample(freq).agg({'qty': np.sum})
    sell_txs = sell_txs.resample(freq).agg({'qty': np.sum})

    buy_txs = buy_txs.rename(columns={"qty": "buy_qty"})
    sell_txs = sell_txs.rename(columns={"qty": "sell_qty"})

    txs = buy_txs.join(sell_txs, on="time", how="outer")
    txs = txs.sort_values(by="time")

    txs['buy_qty'] = txs['buy_qty'].fillna(0)
    txs['sell_qty'] = txs['sell_qty'].fillna(0)
    txs['qty'] = txs['buy_qty'] + txs['sell_qty']
    return txs


def extract_agent_type_actions(parsed_logs_df, agent_type):
    agent_actions = parsed_logs_df.loc[parsed_logs_df['agent_type'] == agent_type, :].copy()
    agent_actions = agent_actions.loc[
        agent_actions['EventType'] == "ORDER_EXECUTED", ["EventTime", "EventType", 'agent_id', 'agent_type',
                                                         "time_placed", "quantity", "side", "order_id", "fill_price",
                                                         'limit_price']].copy()
    agent_actions = agent_actions.rename(columns={"EventTime": "time", "EventType": "event_type"})
    agent_actions['time'] = pd.to_datetime(agent_actions['time'])
    return agent_actions


extract_informer_actions = partial(extract_agent_type_actions, agent_type="InformedAgent")
extract_follower_actions = partial(extract_agent_type_actions, agent_type="FollowerAgent")


def extract_base_informer_stats(end_state):
    informer_ids = [agent.id for agent in iter(end_state['agents']) if agent.type == "InformedAgent"]
    data = []
    for agent_id in iter(informer_ids):
        agent = end_state['agents'][agent_id]
        recommendations_sent = agent.signals_sent
        last_pred = agent.last_pred
        max_quantity = agent.max_units_count
        record = {"agent_id": agent_id, "recommendations_sent": recommendations_sent, "last_pred": last_pred,
                  "max_quantity": max_quantity}
        data.append(record)
    return pd.DataFrame.from_records(data)


def extract_from_paradict_output(output, symbol):
    output_arr = output.split(",")
    symbol_string = None
    for s in output_arr:
        if symbol in s:
            symbol_string = s
            break
    if symbol_string is None:
        return 0
    symbol_val = int(symbol_string.split(":")[1].split()[0].strip())
    return symbol_val


def extract_holdings_updates(parsed_logs_df, symbol="ABM"):
    holdings_df = parsed_logs_df.loc[
        parsed_logs_df['EventType'].isin(['STARTING_CASH', "HOLDINGS_UPDATED", "FINAL_HOLDINGS"]), ["EventTime",
                                                                                                    "EventType",
                                                                                                    "ScalarEventValue",
                                                                                                    "agent_id",
                                                                                                    "agent_type",
                                                                                                    "CASH",
                                                                                                    symbol]].copy()
    holdings_df['holdings_cash'] = holdings_df['CASH']
    holdings_df.loc[holdings_df['EventType'] == "STARTING_CASH", 'holdings_cash'] = holdings_df.loc[
        holdings_df['EventType'] == "STARTING_CASH", 'ScalarEventValue']
    holdings_df.loc[holdings_df['EventType'] == "FINAL_HOLDINGS", 'holdings_cash'] = holdings_df.loc[
        holdings_df['EventType'] == "FINAL_HOLDINGS", 'ScalarEventValue'].apply(
        lambda x: extract_from_paradict_output(x, 'CASH'))

    holdings_df[f'holdings_{symbol}'] = holdings_df[symbol]
    holdings_df.loc[holdings_df['EventType'] == "STARTING_CASH", f'holdings_{symbol}'] = 0
    holdings_df.loc[holdings_df['EventType'] == "FINAL_HOLDINGS", f'holdings_{symbol}'] = holdings_df.loc[
        holdings_df['EventType'] == "FINAL_HOLDINGS", 'ScalarEventValue'].apply(
        lambda x: extract_from_paradict_output(x, symbol))

    holdings_df['holdings_cash'] = holdings_df['holdings_cash'].fillna(0)
    holdings_df[f'holdings_{symbol}'] = holdings_df[f'holdings_{symbol}'].fillna(0)
    holdings_df = holdings_df.rename(columns={"EventTime": "event_time", "EventType": "event_type"})
    holdings_df = holdings_df.drop(columns=["CASH", symbol, "ScalarEventValue"])
    return holdings_df


def extract_surplus(parsed_logs_df):
    cash_df = parsed_logs_df.loc[
        parsed_logs_df['EventType'].isin(['STARTING_CASH', "ENDING_CASH"]), ["EventType",
                                                                             "ScalarEventValue",
                                                                             "agent_id", "agent_type"]].copy()
    cash_df = cash_df.reset_index(drop=True)
    cash_df = cash_df.pivot(columns="EventType", values="ScalarEventValue", index=['agent_id', 'agent_type'])
    cash_df = cash_df.rename(
        columns={"EventTime": "event_time", "EventType": "event_type", "STARTING_CASH": "starting_cash",
                 "ENDING_CASH": "ending_cash",
                 "ScalarEventValue": "cash"})

    cash_df['surplus'] = cash_df['ending_cash'] - cash_df['starting_cash']
    return cash_df
