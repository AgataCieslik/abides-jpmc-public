from .agent import Agent
from typing import Any, List, Optional, Tuple
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
