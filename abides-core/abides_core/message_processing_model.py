from abc import ABC, abstractmethod
from message import Message
from typing import Optional
from . import NanosecondTime
import queue


class MessageProcessingModel(ABC):
    @abstractmethod
    def on_receive(self, current_time: NanosecondTime, message: Message) -> None:
        pass

    @abstractmethod
    def on_decision(self) -> Message:
        pass


class MostRecentProcessingModel(MessageProcessingModel):
    def __init__(self) -> None:
        self.last_message = None
        self.last_message_time = None

    def on_receive(self, current_time: NanosecondTime, message: Message) -> None:
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


class VotingProcessingModel(MessageProcessingModel):
    def __init__(self, max_sample_size: Optional[int] = None) -> None:
        self.max_sample_size = max_sample_size
        self.messages_queue = queue.PriorityQueue[Message] = queue.PriorityQueue()

    def on_receive(self, current_time: NanosecondTime, message: Message) -> None:
        self.messages.put((current_time, Message))

    def on_decision(self) -> Message:
        sample = []
        message_occurencies = {}

        while len(sample) <= self.max_sample_size:
            message = self.messages_queue.get()
            message_type = message.type()
            if message_type not in message_occurencies.keys():
                message_occurencies[message_type] = 1
            else:
                message_occurencies[message_type] = message_occurencies[message_type] + 1
            sample.append(message)
        return max(sample, key=sample.get)

