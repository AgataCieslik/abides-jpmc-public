import rmsc04
from abides_core.network import Network

def build_config(network: Network, **rmsc04_optional_kwargs):
    config = rmsc04.build_config(**rmsc04_optional_kwargs)
    config['agents'].extend(network.agents)
    return config
