from .agent import Agent
from typing import List, Optional
import numpy as np
from .message import Message


class CommunicativeAgent(Agent):
    def __init__(
            self,
            id: int,
            name: Optional[str] = None,
            type: Optional[str] = None,
            random_state: Optional[np.random.RandomState] = None,
            log_events: bool = True,
            log_to_file: bool = True,
            # weźmiemy ich po id, bo tak realizowane jest send_message, może adekwatniejsza nazwa neighbours?
            contacts: List[int] = None,
            # lista opóźnień (by zindywidualizować każdy z kontaktów)
            delays: List[int] = None
    ) -> None:
        super().__init__(id, name, type, random_state, log_events, log_to_file)
        self.contacts = contacts
        if delays is None:
            self.delays = [0] * len(self.contacts)
        else:
            self.delays = delays

    # nazwa na razie umowna - chodzi o wysłanie wiadomości do wszystkich kontaktów
    def broadcast(self, message: Message) -> None:
        # pzypadek gdy lista pusta
        for contact_id, delay in zip(self.contacts, self.delays):
            self.send_message(recipient_id=contact_id, message=message, delay=delay)

    def delete_contact(self, contact_id: int) -> None:
        contact_index = self.contacts.index(contact_id)
        self.contacts.pop(contact_index)
        self.contacts.pop(contact_index)

    def add_contact(self, contact_id: int, delay: Optional[int] = 0) -> None:
        self.contacts.append(contact_id)
        self.delays.append(delay)

    def update_contacts(self, new_contact_list: List[int], delays_list: List[int] = None) -> None:
        self.contacts = new_contact_list
        if delays_list is None:
            self.delays = [0] * len(self.contacts)
        else:
            self.delays = delays_list
