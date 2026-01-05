
import logging

class SlippageModel:
    def __init__(self, base_spread_bps=5, volume_impact_factor=0.001):
        """
        Initializes the slippage model.

        :param base_spread_bps: Baseline bid-ask spread in basis points (bps), default 5 bps = 0.05%
        :param volume_impact_factor: Slippage impact per percentage of volume traded, e.g., 0.001 = 0.1% per 10% volume
        """
        self.base_spread_bps = base_spread_bps
        self.volume_impact_factor = volume_impact_factor
        self._warned_zero_volume = False

    def estimate_slippage(self, current_price, order_size, bar_volume):
        """
        Estimates slippage-adjusted execution price for market orders.

        :param current_price: Last traded or midpoint price
        :param order_size: Quantity of the order to be filled
        :param bar_volume: Total volume traded in the bar interval (e.g., 1 minute)
        :return: Estimated fill price
        """
        if not bar_volume or bar_volume <= 0:
            if not self._warned_zero_volume:
                logging.warning("Bar volume is zero or missing; using base spread only for slippage.")
                self._warned_zero_volume = True
            volume_ratio = 0.0
        else:
            volume_ratio = min(order_size / bar_volume, 1.0)

        slippage_pct = (self.base_spread_bps / 10000.0) + (volume_ratio * self.volume_impact_factor)
        adjusted_price = current_price * (1 + slippage_pct)

        logging.debug(f"Slippage: price={current_price}, size={order_size}, vol={bar_volume}, slip={slippage_pct:.4%}")
        return adjusted_price

    def simulate_fill(self, current_price, order_size, bar_volume):
        """
        Simulates a filled order price considering slippage for backtesting.

        :param current_price: Market price from historical data
        :param order_size: Order size in units
        :param bar_volume: Historical volume during this interval
        :return: Fill price
        """
        return self.estimate_slippage(current_price, order_size, bar_volume)

    def estimate_cost(self, order_size, bar_volume, current_price):
        """
        Estimates transaction cost in absolute dollars from slippage.

        :param order_size: Quantity ordered
        :param bar_volume: Volume of the time window
        :param current_price: Current price
        :return: Estimated total cost from slippage
        """
        fill_price = self.simulate_fill(current_price, order_size, bar_volume)
        cost = (fill_price - current_price) * order_size
        return max(cost, 0.0)
