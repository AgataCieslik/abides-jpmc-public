# jak ma działać funkcja
# parametry:
# parametry wyroczni
# parametry informed agenta
# parametry followerów, w szczególności czasy ich przybycia
# 1: generuje standardowe rmsc + oracle ze specjalnym eventem
# 2: generuje informed agent wraz z ustalonymi parametrami oraz followerów
# 3: przygotować kilka gotowych profili przy pomocy partial, gdzie tylko zmieniamy pojedyncze paametry

from .rmsc04 import build_config as build_rmsc04_config
from datetime import datetime
from abides_markets.oracles import SpecialEventOracle
from abides_markets.utils import generate_latency_model
from abides_core.network import CentralizedNetwork
from abides_markets.agents import InformedAgent, FollowerAgent
from itertools import count
from abides_core.utils import get_wake_time, str_to_ns
from abides_markets.models import OrderSizeModel
import pandas as pd


# do rozważenia: wrzucić to do utils
def generate_informed_flow_agents(followers_num, id_generator, informer_params, follower_params, wake_time_func,
                                  wake_time_func_params):
    # generujemy agentów do sieci
    informer = InformedAgent(id=next(id_generator), **informer_params)
    followers = [
        FollowerAgent(id=next(id_generator), wakeup_time=wake_time_func(**wake_time_func_params), **follower_params) for
        i in range(followers_num)]
    graph = CentralizedNetwork.construct_from_agent_list(central_agent=informer, agent_list=followers)
    return graph.get_agents()


def build_config(
        seed=int(datetime.now().timestamp() * 1_000_000) % (2 ** 32 - 1),
        date="20210205",
        end_time="16:00:00",
        stdout_log_level="INFO",
        ticker="ABM",
        starting_cash=10_000_000,  # Cash in this simulator is always in CENTS.
        log_orders=True,
        # oracle
        oracle_type=SpecialEventOracle,
        kappa_oracle=1.67e-16,  # Mean-reversion of fundamental time series.
        sigma_s=0,
        fund_vol=5e-5,  # Volatility of fundamental time series (std).
        megashock_lambda_a=2.77778e-18,
        megashock_mean=0,
        megashock_var=0,
        # zakładam że może być tylko jedno wydarzenie
        special_event=None,
        optional_oracle_kwargs={},
        # zakładamy max jednego informera
        include_informer=True,
        informed_agent_confidence=0.2,
        informed_agent_kwargs={},
        # liczba followerów
        num_followers=0,
        follower_agent_kwargs={"order_size_model": OrderSizeModel()},
        follower_wake_time_func=get_wake_time,
        follower_wake_time_params={}):
    # dopusczamy jedynie możliwość modyfikowania podstawowych parametrów i wyroczni
    if special_event is not None:
        optional_oracle_kwargs['special_events'] = {ticker: [special_event]}
    config = build_rmsc04_config(seed=seed, date=date, end_time=end_time, stdout_log_level=stdout_log_level,
                                 ticker=ticker, starting_cash=starting_cash, log_orders=log_orders,
                                 oracle_type=oracle_type, kappa_oracle=kappa_oracle, sigma_s=sigma_s,
                                 fund_vol=fund_vol, megashock_lambda_a=megashock_lambda_a,
                                 megashock_mean=megashock_mean, megashock_var=megashock_var,
                                 optional_oracle_kwargs=optional_oracle_kwargs)

    if include_informer == True:
        if special_event is not None:
            mean_pv = special_event['SpecialEventValue']
            time_horizon = special_event['SpecialEventTime']
            if "time_horizon" not in informed_agent_kwargs.keys():
                informed_agent_kwargs['time_horizon'] = time_horizon
            informed_agent_kwargs['mean_pv'] = mean_pv
            if "sigma_pv" not in informed_agent_kwargs.keys():
                sigma_pv = informed_agent_confidence * mean_pv
                informed_agent_kwargs['sigma_pv'] = sigma_pv

        agents_num = len(config['agents'])
        total_agents_num = agents_num + 1 + num_followers
        id_generator = count(agents_num)

        if "open_time" not in follower_wake_time_params.keys():
            # noise_open = int(pd.to_datetime(date).to_datetime64()) + str_to_ns("09:30:00") - str_to_ns("00:30:00")
            noise_open = int(pd.to_datetime(date).to_datetime64()) + str_to_ns("09:30:00")
            follower_wake_time_params['open_time'] = noise_open
        if "close_time" not in follower_wake_time_params.keys():
            # follower_wake_time_params['close_time'] = time_horizon + str_to_ns("00:30:00")
            follower_wake_time_params['close_time'] = time_horizon
        if "time_horizon" not in follower_agent_kwargs.keys():
            # zakładam, że efekt wydarzenia może nie być widoczny od razu
            follower_agent_kwargs['time_horizon'] = time_horizon
            # follower_agent_kwargs['time_horizon'] = time_horizon + str_to_ns("00:30:00")

        informed_flow_agents = generate_informed_flow_agents(num_followers, id_generator, informed_agent_kwargs,
                                                             follower_agent_kwargs,
                                                             wake_time_func=follower_wake_time_func,
                                                             wake_time_func_params=follower_wake_time_params)
        # tu dokładamy nowych agentów
        config['agents'].extend(informed_flow_agents)

        # aktualizujemy model latency
        config['agent_latency_model'] = generate_latency_model(total_agents_num)
    return config
