from abides_core import NanosecondTime


class Oracle:
    def get_daily_open_price(
        self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int:
        raise NotImplementedError
# dopisać metodę , która dla danych: progu procentowego i horyzontu czasowego zwraca sygnał buy/sell/hold (dla przyszłości,
# na podstawie wartości fundamentalnej) - nie będziemy tego robić