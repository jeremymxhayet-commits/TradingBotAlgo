from strategies.base_strategy import Strategy


class StatisticalArbitrageStrategy(Strategy):
    def __init__(self, symbols):
        super().__init__()
        self.symbols = symbols

    def compute_signals(self, data):
        signals = []
        for symbol in self.symbols:
            signals.append({"symbol": symbol, "action": "hold"})
        return signals

    def generate_orders(self, signals):
        return []
