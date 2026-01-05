import asyncio
import logging
from datetime import datetime


from config import (
    TRADE_UNIVERSE,
    DEFAULT_BAR_INTERVAL,
    LIVE_TRADING,
    REFRESH_INTERVAL,
    LOG_PATH,
    ENABLE_HEDGING
)


from data_ingestion.live_feed import LiveFeed
from execution.order_manager import OrderManager
from strategies.ml_strategy import MLStrategy
from portfolio.risk_management import RiskManager



import os
os.makedirs(LOG_PATH, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_PATH, f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)
print("Logging initialized.")




async def run_live_trading():
    print("Initializing Live Trading Environment...")
    logger.info("Initializing Live Trading...")


    order_manager = OrderManager(live=LIVE_TRADING)
    await order_manager.connect()

 
    symbols = TRADE_UNIVERSE["equities"]  
    feed = LiveFeed(symbols=symbols, interval=DEFAULT_BAR_INTERVAL)
    await feed.connect()

   
    strategy = MLStrategy(symbols)
    risk_manager = RiskManager()

    logger.info(f"Strategy initialized with universe: {symbols}")

    try:
        while True:
          
            data = await feed.get_next_bar()

            if data is None:
                logger.warning("No new data received from live feed.")
                await asyncio.sleep(REFRESH_INTERVAL.total_seconds())
                continue

            logger.info(f"Received new market data for {list(data.keys())}")

          
            signals = strategy.compute_signals(data)
            logger.info(f"Signals: {signals}")

          
            approved_trades = risk_manager.filter_trades(signals)
            logger.info(f"Approved trades: {approved_trades}")

         
            for trade in approved_trades:
                order = order_manager.build_order(trade)
                response = await order_manager.execute_order(order)
                logger.info(f"Executed order: {response}")

            
            if ENABLE_HEDGING:
                hedge_orders = risk_manager.generate_hedge_orders()
                for hedge in hedge_orders:
                    response = await order_manager.execute_order(hedge)
                    logger.info(f"Hedging order executed: {response}")

           
            await asyncio.sleep(REFRESH_INTERVAL.total_seconds())

    except KeyboardInterrupt:
        logger.warning("Manual shutdown initiated.")
    finally:
        await order_manager.disconnect()
        await feed.disconnect()
        logger.info("Live trading stopped.")




if __name__ == "__main__":
    print("Launching live trading bot...")
    try:
        asyncio.run(run_live_trading())
    except Exception as e:
        logger.exception(f"Fatal error in live trading loop: {e}")
