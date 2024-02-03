from abc import ABC
from dataclasses import dataclass

from abides_core import Message

@dataclass
class TradingSignal(Message,ABC):
    symbol: str

@dataclass
class BuySignal(TradingSignal):
    pass

@dataclass
class SellSignal(TradingSignal):
    pass

