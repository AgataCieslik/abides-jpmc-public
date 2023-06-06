from .communicative_agent import CommunicativeAgent
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Generator
from typing_extensions import Self
import numpy as np


# czy klasa z abstrakcyjnymi metodami musi dziedziczyć po ABC? chyba nie?
class Network:
    def __init__(self, agents: List[CommunicativeAgent] = None) -> None:
        self.agents = agents

    @classmethod
    @abstractmethod
    def construct_from_agent_list(cls, agent_list: List[CommunicativeAgent]) -> Self:
        pass

    @property
    def size(self):
        return len(self.agents)

    @classmethod
    def construct(cls, agent_types: Dict[Type[CommunicativeAgent], int],
                  id_generator: Generator[int, None, None]) -> Self:
        agents_in_net = []
        for agent_type, agent_count in agent_types.items():
            for i in range(agent_count):
                agents_in_net.append(agent_type(id=next(id_generator)))
        return cls.construct_from_agent_list(agents_in_net)

    def get_agent_list(self) -> List[CommunicativeAgent]:
        return self.agents

    def get_agent_with_id(self, id: int) -> CommunicativeAgent:
        for i, agent in enumerate(self.agents):
            if agent.id == id:
                return agent

    def get_friends(self, agent_id: int) -> List[CommunicativeAgent]:
        agent = self.get_agent_with_id(agent_id)
        return agent.contacts

    def get_agent_degree(self, agent_id: int) -> int:
        if self.get_friends(agent_id) is not None:
            return len(self.get_friends(agent_id))
        return 0

    # te metody są niesymetryczne
    def set_connection(self, agent0_id: int, agent1_id: int) -> None:
        agent0 = self.get_agent_with_id(agent0_id)
        agent0.add_contact(agent1_id)

    def remove_connection(self, agent0_id: int, agent1_id: int) -> None:
        agent0 = self.get_agent_with_d(agent0_id)
        agent0.delete_contact(agent1_id)

    def add_agent(self, agent: CommunicativeAgent) -> None:
        self.agents.append(agent)


class CompleteGraph(Network):
    @classmethod
    def construct_from_agent_list(cls, agent_list: List[CommunicativeAgent]) -> Self:
        agents_ids = [agent.id for agent in iter(agent_list)]
        for agent in iter(agent_list):
            new_contact_list = [agent_id for agent_id in iter(agents_ids) if agent_id != agent.id]
            agent.update_contacts(new_contact_list=new_contact_list)
        return cls(agent_list)


class CentralizedNetwork(Network):
    def __init__(self, central_agent: CommunicativeAgent = None, agents: List[CommunicativeAgent] = None) -> None:
        super().__init__(agents)
        self.agents.append(central_agent)
        self.central_agent: CommunicativeAgent = central_agent
        # tak długo jak id musi odpowiadać indeksowi w tablicy agentów kernela, musimy tą tablicę przesortować
        # TODO: zastanowić się, czy to sortowanie robić tu, czy już "na zewnątrz"
        self.agents.sort(key=lambda agent: agent.id)

    @classmethod
    def construct_from_agent_list(cls, central_agent: CommunicativeAgent, agent_list: List[CommunicativeAgent]) -> Self:
        agents_ids = [agent.id for agent in iter(agent_list)]
        new_contact_list = [agent_id for agent_id in iter(agents_ids) if agent_id != central_agent.id]
        central_agent.update_contacts(new_contact_list=new_contact_list)
        for agent in iter(agent_list):
            agent.update_contacts(new_contact_list=[central_agent.id])
        return cls(central_agent, agent_list)

    @classmethod
    def construct(cls, central_agent_type: Type[CommunicativeAgent], agent_types: Dict[Type[CommunicativeAgent], int],
                  id_generator: Generator[int, None, None]) -> Self:
        agents_in_net = []
        central_agent = central_agent_type(id=next(id_generator))
        for agent_type, agent_count in agent_types.items():
            for i in range(agent_count):
                agents_in_net.append(agent_type(id=next(id_generator)))
        return cls.construct_from_agent_list(central_agent, agents_in_net)


class BANetwork(Network):
    def __init__(self, starting_state: Network, random_state: np.random.RandomState, m: int,
                 agents: List[CommunicativeAgent]) -> None:
        super().__init__(agents)
        self.random_state: np.random.RandomState = (
                random_state
                or np.random.RandomState(
            seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
        )
        )
        self.m = m
        self.starting_state = starting_state

    @classmethod
    def construct_from_agent_list(cls, starting_state: Network, new_agents: List[CommunicativeAgent], m: int,
                                  random_state: np.random.RandomState = None) -> Self:

        net = cls(starting_state, random_state, m, starting_state.agents)

        for new_agent in iter(new_agents):
            agents_ids = [agent.id for agent in iter(net.agents)]
            K = [net.get_agent_degree(agent_id) for agent_id in iter(agents_ids)]
            K_sum = sum(K)
            probs = [k / K_sum for k in iter(K)]

            choice = net.random_state.choice(a=agents_ids, size=m, p=probs, replace=False)
            net.agents.append(new_agent)

            for agent_id in iter(choice):
                net.set_connection(agent_id, new_agent.id)
                net.set_connection(new_agent.id, agent_id)

        return net

    @classmethod
    def construct(cls, starting_state: Network, agent_types: Dict[Type[CommunicativeAgent], int], m: int,
                  id_generator: Generator[int, None, None], random_state: np.random.RandomState = None) -> Self:
        new_agents = []
        for agent_type, agent_count in agent_types.items():
            for i in range(agent_count):
                new_agents.append(agent_type(id=next(id_generator)))
        return cls.construct_from_agent_list(starting_state, new_agents, m, random_state)


class ERNetwork(Network):
    def __init__(self, agents: List[CommunicativeAgent], p: float, random_state: np.random.RandomState):
        super().__init__(agents)
        self.p = p
        self.random_state: np.random.RandomState = (
                random_state
                or np.random.RandomState(
            seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
        )
        )

    @classmethod
    def construct_from_agent_list(cls, agents: List[CommunicativeAgent], p: float, random_state: np.random.RandomState):
        net = cls(agents, p, random_state)
        agents_ids = [agent.id for agent in iter(net.agents)]
        possible_edges = [(agent0, agent1) for agent0_ind, agent0 in enumerate(agents_ids) for agent1 in
                          agents_ids[agent0_ind + 1:]]
        choices = []
        for agent0_id, agent1_id in iter(possible_edges):
            choice = net.random_state.choice(a=[False, True], size=1, p=[1 - p, p])[0]
            choices.append(choice)
            if choice:
                net.set_connection(agent0_id, agent1_id)
                net.set_connection(agent1_id, agent0_id)
        return net

    @classmethod
    def construct(cls, agent_types: Dict[Type[CommunicativeAgent], int], p: float,
                  id_generator: Generator[int, None, None], random_state: np.random.RandomState = None) -> Self:
        new_agents = []
        for agent_type, agent_count in agent_types.items():
            for i in range(agent_count):
                new_agents.append(agent_type(id=next(id_generator)))
        return cls.construct_from_agent_list(new_agents, p, random_state)
