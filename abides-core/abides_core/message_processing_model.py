from abc import ABC, abstractmethod
from .message import Message
from typing import Optional, Type, List, Tuple
from . import NanosecondTime
import queue
import numpy as np


class MessageProcessingModel(ABC):
    @abstractmethod
    def on_receive(self, current_time: NanosecondTime, message: Message, sender_id: int) -> None:
        pass

    @abstractmethod
    def on_decision(self) -> Message:
        pass

    @abstractmethod
    def validate(self) -> None:
        pass


class MostRecentProcessingModel(MessageProcessingModel):
    def __init__(self) -> None:
        self.last_message = None
        self.last_message_time = None

    def on_receive(self, current_time: NanosecondTime, message: Message, sender_id: int) -> None:
        if self.last_message is not None and self.last_message_time is not None:
            if current_time < self.last_message_time:
                return
        else:
            self.last_message = message
            self.last_message_time

    def on_decision(self) -> Message:
        most_recent_message = self.last_message
        self.last_message = None
        self.last_message_time = None
        return most_recent_message

    def validate(self) -> None:
        pass


class VotingProcessingModel(MessageProcessingModel):
    def __init__(self, max_sample_size: Optional[int] = np.inf) -> None:
        self.max_sample_size = max_sample_size
        self.messages_queue: queue.PriorityQueue[Message] = queue.PriorityQueue()

    def on_receive(self, current_time: NanosecondTime, message: Message, sender_id: int) -> None:
        self.messages_queue.put((current_time, message))

    def on_decision(self) -> Message:
        sample_size = 0
        message_occurencies = {}
        message_passed = None

        while (sample_size <= self.max_sample_size) and (not self.messages_queue.empty()):
            # ściągamy ostatnią wiadomość z kolejki i określamy jej typ
            delivery_time, message = self.messages_queue.get()
            message_type = type(message)

            # dodajemy wystąpienie rodzaju wiadomości do słownika
            if message_type not in message_occurencies.keys():
                message_occurencies[message_type] = 1
            else:
                message_occurencies[message_type] = message_occurencies[message_type] + 1

            # jeśli wiadomość jest ostatnią wiadomością najcześciej występującego typu, zapisujemy ją jako wiadomość do przekazania
            if max(message_occurencies, key=message_occurencies.get) == message_type:
                message_passed = message
        return message_passed

    def validate(self) -> None:
        pass


class ProbabilisticProcessing(MessageProcessingModel):
    # jako prior belief bierzemy parametry rozkładu beta: alfa i beta (dla alfa = beta =1 dostajemy rozklad jednostajny)
    def __init__(self, message_types: List[Type[Message]], contacts: List[int],
                 prior_belief: Tuple[float, float] = (0, 0)) -> None:
        self.messages_queue: queue.PriorityQueue[Message] = queue.PriorityQueue()
        self.decisions_count = 0
        self.contacts = contacts
        # rozkład prawdopodobieństwa zmiennej prawdomówności/uczciwości
        self.alpha, self.beta = prior_belief
        self.true_messages_count = {i: 0 for i in range(len(contacts))}
        self.messages_to_validate = []

    def on_receive(self, current_time: NanosecondTime, message: Message, sender_id: int) -> None:
        self.messages_queue.put((current_time, (sender_id, Message)))

    def validate(self, truth: Type[Message]) -> None:
        for sender_id, message in iter(self.messages_to_validate):
            if isinstance(message, truth):
                self.true_messages_count += 1
        self.messages_to_validate = []

    # może jako property?
    def trust(self) -> List[float]:
        return {agent_id: (true_responses + self.alpha - 1) / (self.decisions_count + self.alpha + self.beta - 2) for
                agent_id, true_responses in self.true_messages_count.items()}


class TrustWeightedProcessing(ProbabilisticProcessing):
    def on_decision(self) -> Message:
        messages_by_sender = {}
        message_occurencies = {}
        message_passed = None
        current_trust = self.trust()

        while not self.message_queue.empty():
            # ściągamy ostatnią wiadomość z kolejki i określamy jej typ
            delivery_time, message_with_sender = self.messages_queue.get()
            sender_id, message = message_with_sender
            messages_by_sender[sender_id] = message

            message_type = type(message)

            # dodajemy wystąpienie rodzaju wiadomości * zaufanie do nadawcy do słownika
            if message_type not in message_occurencies.keys():
                message_occurencies[message_type] = 1 * current_trust[sender_id]
            else:
                message_occurencies[message_type] = message_occurencies[message_type] + 1 * current_trust[sender_id]

            # jeśli wiadomość jest ostatnią wiadomością najcześciej występującego typu, zapisujemy ją jako wiadomość do przekazania
            if max(message_occurencies, key=message_occurencies.get) == message_type:
                message_passed = message
        self.messages_to_validate = [(sender_id, message) for sender_id, message in messages_by_sender.items()]
        return message_passed


class TrustedSourceProcessing(ProbabilisticProcessing):
    def on_decision(self) -> Message:
        messages_by_sender = {}
        current_trust = self.trust()

        while not self.message_queue.empty():
            delivery_time, message_with_sender = self.messages_queue.get()
            sender_id, message = message_with_sender
            messages_by_sender[sender_id] = message
        trust_in_sample = {sender: sender_trust for sender, sender_trust in current_trust.items() if
                           sender in messages_by_sender.keys()}
        trusted_sender = max(trust_in_sample, key=trust_in_sample.get)
        self.messages_to_validate = [(sender_id, message) for sender_id, message in messages_by_sender.items()]
        return messages_by_sender[trusted_sender]
