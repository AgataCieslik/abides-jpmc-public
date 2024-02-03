from abc import ABC
from dataclasses import dataclass
from abides_core import NanosecondTime
from abides_core import Message


@dataclass
#TODO: tą opcję trzeba wykluczyć
class QueryFinalValue(Message):
    '''
    Message sent to alleged insider agent in order to obtain his fundamental value estimate and its variance estimate.
    '''
    symbol: str


@dataclass
class FinalValueResponse(Message):
    '''
    Response for FundamentalValueQuery. Contains symbol, timestamp and fundamental value and variance estimates.
    '''
    symbol: str
    obs_time: NanosecondTime
    r_T: float
    sigma_t: float


@dataclass
class QuerySideRecommendation(Message):
    '''
    Message sent to alleged insider agent in order to obtain BUY/SELL recommendation.
    '''
    symbol: str
