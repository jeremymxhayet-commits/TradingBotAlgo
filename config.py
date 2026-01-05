import os
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()


LIVE_TRADING = False                                                                     


ALPACA_API_KEY = os.getenv("_____________________________")
ALPACA_SECRET_KEY = os.getenv("_________________________________")
ALPACA_BASE_URL = (
    "https://api.alpaca.markets" if LIVE_TRADING else "https://paper-api.alpaca.markets"
)
ALPACA_DATA_URL = "https://data.alpaca.markets"

ALPACA_STREAM_URL = (
    "wss://api.alpaca.markets/stream" if LIVE_TRADING else "wss://paper-api.alpaca.markets/stream"
)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")
OANDA_BASE_URL = os.getenv(
    "OANDA_BASE_URL",
    "https://api-fxpractice.oanda.com" if OANDA_ENV == "practice" else "https://api-fxtrade.oanda.com",
)


TRADE_UNIVERSE = {
    "equities": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA", "BRK-B", "JPM",
        "V", "MA", "UNH", "XOM", "JNJ", "PG", "HD", "BAC", "AVGO", "PFE",
        "KO", "PEP", "TMO", "ABBV", "COST", "CRM", "ACN", "CSCO", "ORCL", "INTC",
        "WMT", "MCD", "NKE", "DIS", "NFLX", "AMD", "QCOM", "TXN", "ADP", "LIN",
        "LOW", "UPS", "CAT", "GE", "IBM", "CVX", "MRK", "ABT", "DHR", "BMY",
        "WFC", "GS", "MS", "C", "BA", "AMGN", "MDT", "SBUX", "AMAT", "ISRG",
        "GILD", "LLY", "VZ", "SPGI", "BLK", "INTU", "NOW", "PANW", "MU", "BKNG",
        "TSM", "LRCX", "KLAC", "MDLZ", "USB", "SCHW", "DE", "F", "GM", "FDX",
        "RTX", "LMT", "NOC", "GD", "MMM", "HON", "UNP", "NSC", "CSX", "DAL",
        "UAL", "AAL", "LUV", "EBAY", "ROKU", "SNAP", "TWLO", "SHOP", "SQ", "PYPL",
        "ABNB", "UBER", "LYFT", "ZM", "DOCU", "TEAM", "OKTA", "ZS", "CRWD", "DDOG",
        "NET", "SNOW", "MDB", "PLTR", "PINS", "ETSY", "TTD", "SPOT", "RBLX", "COIN",
        "HOOD", "MRNA", "BNTX", "REGN", "VRTX", "BIIB", "ILMN", "DXCM", "ZTS", "EA",
        "TTWO", "CMCSA", "TMUS", "TGT", "CME", "ICE", "BK", "PNC", "TFC", "AXP",
        "COF", "AIG", "PRU", "MET", "TRV", "ALL", "AFL", "CB", "PGR", "MMC",
        "AJG", "WTW", "MO", "PM", "EL", "CL", "KMB", "GIS", "K", "KHC",
        "SYY", "KR", "DG", "DLTR", "ROST", "TJX", "ORLY", "AZO", "TSN", "COP",
        "EOG", "SLB", "HAL", "PSX", "MPC", "VLO", "OXY", "DVN", "KMI", "NEE",
        "DUK", "SO", "D", "EXC", "AEP", "SRE", "XEL", "ED", "PEG", "PLD",
        "AMT", "CCI", "EQIX", "SPG", "O", "PSA", "WELL", "DLR", "VTR", "RSG",
    ],
    "etfs": [],
    "crypto": [
        "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "XRP/USD",
        "BNB/USD", "DOGE/USD", "AVAX/USD", "DOT/USD", "LTC/USD",
    ],
    "forex": [],
    "options": [],
}

DEFAULT_BAR_INTERVAL = "1Min"                                              

MARKET_OPEN_TIME = "09:30"
MARKET_CLOSE_TIME = "16:00"


MAX_CAPITAL_ALLOCATED = 0.8                                              
MAX_POSITION_PER_ASSET = 0.3                            

MAX_DRAWDOWN_PCT = 0.25                                                     
STOP_LOSS_PCT = 0.25                        
TAKE_PROFIT_PCT = 0.12                              
TRAILING_STOP_PCT = 0.10                                             
VOLATILITY_TARGET = 0.015                                       

RISK_PER_TRADE = 0.02                                  

ENABLE_HEDGING = True

ENABLED_ASSET_CLASSES = ["equities", "crypto"]

YAHOO_SYMBOL_MAP = {
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "SOL/USD": "SOL-USD",
    "ADA/USD": "ADA-USD",
    "XRP/USD": "XRP-USD",
    "BNB/USD": "BNB-USD",
    "DOGE/USD": "DOGE-USD",
    "AVAX/USD": "AVAX-USD",
    "DOT/USD": "DOT-USD",
    "LTC/USD": "LTC-USD",
}


XGB_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "verbosity": 0,
}

LSTM_PARAMS = {
    "sequence_length": 60,
    "epochs": 10,
    "batch_size": 32,
    "units": 64,
    "dropout": 0.2,
    "recurrent_dropout": 0.1
}

RL_PARAMS = {
    "algo": "PPO",
    "env_id": "TradingEnv-v0",
    "total_timesteps": 100_000,
    "policy_kwargs": {"net_arch": [64, 64]}
}

ML_THRESHOLD = 0.40
ML_ONLINE_LEARNING = True
ML_ONLINE_WINDOW = 252
ML_ONLINE_INTERVAL = 5
ML_ONLINE_MIN_SAMPLES = 200
ASSET_CLASS_THRESHOLDS = {
    "equities": 0.40,
    "etfs": 0.40,
    "crypto": 0.55,
    "forex": 0.55,
    "options": 0.60,
}

NO_TRADE_PROB_BAND = 0.03                                          
MIN_TREND_STRENGTH = 0.00025                                    
MIN_VOLATILITY = 0.002                                         
VOL_LOOKBACK = 20
ATR_LOOKBACK = 14
TREND_LOOKBACK = 20
MIN_EDGE_TO_COST = 0.75                                                    
MAX_VOLUME_PARTICIPATION = 0.05                           
MIN_TRADE_QTY = 1

ATR_STOP_MULT = 2.5
ATR_TAKE_PROFIT_MULT = 4.0
ATR_TRAIL_MULT = 3.0

ENSEMBLE_LOOKBACK = 30
ENSEMBLE_MIN_WEIGHT = 0.05
ENSEMBLE_MIN_SCORE = 0.08

ALLOW_SHORT = True


BACKTEST_START_DATE = "2020-01-02"
BACKTEST_END_DATE = "2020-12-30"
TRAIN_START_DATE = "2005-01-01"
TRAIN_END_DATE = "2019-12-31"
VALIDATION_START_DATE = "2016-01-01"
VALIDATION_END_DATE = "2017-12-31"
TEST_START_DATE = "2018-01-01"
TEST_END_DATE = "2019-12-31"
WALK_FORWARD_TEST_MONTHS = 6
WALK_FORWARD_RETRAIN_MONTHS = 6
WALK_FORWARD_EXPANDING = True
INITIAL_CAPITAL = 100_000
SLIPPAGE = 0.0005                            
COMMISSION = 0.0001                    


DATA_PATH = "data/"
MODEL_PATH = "models/trained/"
LOG_PATH = "logs/"
PLOTS_PATH = "plots/"
RESULTS_PATH = "backtesting/results/"

XGB_MODEL_PATH = os.path.join(MODEL_PATH, "xgb_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_PATH, "lstm_model.keras")
RL_MODEL_PATH = os.path.join(MODEL_PATH, "rl_agent.zip")
XGB_CLASS_MODEL_PATHS = {
    "equities": os.path.join(MODEL_PATH, "xgb_equities.pkl"),
    "etfs": os.path.join(MODEL_PATH, "xgb_etfs.pkl"),
    "crypto": os.path.join(MODEL_PATH, "xgb_crypto.pkl"),
    "forex": os.path.join(MODEL_PATH, "xgb_forex.pkl"),
    "options": os.path.join(MODEL_PATH, "xgb_options.pkl"),
}


SEED = 42
TIMEZONE = "America/New_York"
REFRESH_INTERVAL = timedelta(seconds=60)                                       
