from .communicative_agent import CommunicativeAgent
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Self


# czy klasa z abstrakcyjnymi metodami musi dziedziczyć po ABC? chyba nie?
class Network:
    def __init__(self, agents: List[CommunicativeAgent] = None) -> None:
        self.size = len(agents)
        self.agents = agents

    @classmethod
    @abstractmethod
    def construct_from_agent_list(cls, agent_list: List[CommunicativeAgent]) -> Self:
        pass

    @classmethod
    def construct(cls, agent_types: Dict[Type[CommunicativeAgent], int]) -> Self:
        agents_in_net = []
        for agent_type, agent_count in agent_types.items():
            for i in range(agent_count):
                agents_in_net.append(agent_type())
        return cls.construct_from_agent_list(agents_in_net)

    def get_agent_list(self) -> List[CommunicativeAgent]:
        return self.agents

    def get_agent_with_id(self, id: int) -> CommunicativeAgent:
        for i, agent in enumerate(self.agents):
            if agent.id == id:
                return agent

    def get_friends(self, agent_id: int) -> List[CommunicativeAgent]:
        agent = self.get_agent_with_id(agent_id)
        return agent.contact

    # te metody są niesymetryczne
    def set_connection(self, agent0_id: int, agent1_id: int) -> None:
        agent0 = self.get_agent_with_id(agent0_id)
        agent0.add_contact(agent1_id)

    def remove_connection(self, agent0_id: int, agent1_id: int) -> None:
        agent0 = self.get_agent_with_d(agent0_id)
        agent0.delete_contact(agent1_id)


class CompleteGraph(Network):
    @classmethod
    def construct_from_agent_list(cls, agent_list: List[CommunicativeAgent]) -> Self:
        agents_ids = [agent.id for agent in iter(agent_list)]
        for agent in iter(agent_list):
            new_contacts_list = [agent_id for agent_id in iter(agents_ids) if agent_id != agent.id]
            agent.update_contacts(new_contacts_list=new_contacts_list)
        return cls(agent_list)

# czy właściwie tego potrzebuję? właściwie to jakiej struktury potrzebuję?
class Ring(Network):
    pass
