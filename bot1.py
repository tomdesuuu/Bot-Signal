# crypto_trading_platform_pro.py
import os
import ccxt
import pandas as pd
import numpy as np
import time
import threading
import pytz
import joblib
import sys
import requests
import hashlib
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from sqlalchemy import create_engine, Column, Float, String, Integer, DateTime, JSON, text, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from telegram import Update
from telegram.ext import CommandHandler, Application, ContextTypes
from flask import Flask, render_template, jsonify, request
import ta  # Technical analysis library
import warnings
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from apscheduler.schedulers.background import BackgroundScheduler
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kaleido  # For static plot export
import sentry_sdk
from scipy.stats import spearmanr


# Initialize Sentry for error tracking
sentry_sdk.init(
    os.getenv('SENTRY_DSN'),
    traces_sample_rate=1.0,
    environment=os.getenv('ENVIRONMENT', 'development'),
    release="crypto-trading-bot@1.4.0"
)

# Fix for Windows encoding issues
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(encoding='utf-8') if hasattr(sys.stderr, 'reconfigure') else None

warnings.filterwarnings('ignore')

# Initialize logging with UTF-8 encoding
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log", encoding='utf-8'),
        UTF8StreamHandler()  # Use our custom handler
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# =====================
# CONFIGURATION - PRODUCTION GRADE (UPDATED)
# =====================
class Config:
    # Focus on major cryptocurrencies with high liquidity
    SYMBOLS = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
        'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT'
    ]
    
    # Timeframes for different trading styles
    TIMEFRAMES = ['15m', '1h', '4h']
    
    # Risk management (updated thresholds)
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.01))
    MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', 0.05))
    SLIPPAGE_PCT = float(os.getenv('SLIPPAGE_PCT', 0.0003))
    FEE_PCT = float(os.getenv('FEE_PCT', 0.0004))
    MAX_LEVERAGE = int(os.getenv('MAX_LEVERAGE', 3))
    
    # Telegram configuration (increased timeout)
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    TELEGRAM_ALLOWED_IDS = set(os.getenv('TELEGRAM_ALLOWED_IDS', '').split(','))
    
    # Model configuration
    MODEL_UPDATE_INTERVAL = int(os.getenv('MODEL_UPDATE_INTERVAL', 24))
    HISTORICAL_DATA_LIMIT = int(os.getenv('HISTORICAL_DATA_LIMIT', 5000))  # Increased from 10000
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
    PORTFOLIO_VALUE = float(os.getenv('PORTFOLIO_VALUE', 10000))
    MODEL_VERSION = "v1.4"
    MIN_TRAINING_DATA = int(os.getenv('MIN_TRAINING_DATA', 500))  # Lowered from 2000
    
    # Hybrid system parameters (relaxed thresholds)
    USE_ML_CONFIRMATION = os.getenv('USE_ML_CONFIRMATION', 'true').lower() == 'true'
    MIN_BREAKOUT_VOLUME_RATIO = float(os.getenv('MIN_BREAKOUT_VOLUME_RATIO', 1.3))
    MAX_VOLATILITY_RATIO = float(os.getenv('MAX_VOLATILITY_RATIO', 0.06))
    
    # Exchange configuration
    EXCHANGE_CONFIG = {
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_API_SECRET'),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True,
            'recvWindow': 10000
        }
    }
    
    # News API (optional)
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    NEWS_SOURCES = ['coindesk', 'cointelegraph', 'cryptonews']
    
    # Performance monitoring
    PERFORMANCE_REPORT_INTERVAL = int(os.getenv('PERFORMANCE_REPORT_INTERVAL', 24))
    
    # Trading hours filter (UTC)
    TRADING_HOURS = {
        'start': int(os.getenv('TRADING_HOURS_START', 0)),  # 0 = 00:00 UTC
        'end': int(os.getenv('TRADING_HOURS_END', 24))      # 24 = 24:00 UTC
    }
    
    # Debug mode (new)
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'saved_models')

# Verify critical configurations
if not Config.TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN not set in environment variables!")
if not Config.TELEGRAM_CHAT_ID:
    logger.error("TELEGRAM_CHAT_ID not set in environment variables!")
if not Config.EXCHANGE_CONFIG['apiKey']:
    logger.error("BINANCE_API_KEY not set in environment variables!")
if not Config.EXCHANGE_CONFIG['secret']:
    logger.error("BINANCE_API_SECRET not set in environment variables!")

# =====================
# DATABASE SETUP
# =====================
Base = declarative_base()

class Portfolio(Base):
    __tablename__ = 'portfolio'
    id = Column(Integer, primary_key=True)
    total_risk = Column(Float, default=0.0)
    total_value = Column(Float, default=Config.PORTFOLIO_VALUE)
    last_updated = Column(DateTime, default=datetime.utcnow)

class Signal(Base):
    __tablename__ = 'signals'
    id = Column(String, primary_key=True)
    symbol = Column(String)
    timeframe = Column(String)
    timestamp = Column(DateTime)
    direction = Column(String)  # LONG/SHORT
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profits = Column(JSON)  # List of TP prices
    confidence = Column(Float)
    market_regime = Column(String)  # TRENDING/RANGING/BREAKOUT
    position_size = Column(Float)  # Position size as % of portfolio
    status = Column(String)  # ACTIVE/CLOSED/CANCELLED
    closed_at = Column(DateTime)
    closed_price = Column(Float)
    outcome = Column(String)  # WIN/LOSS/CANCELLED
    volatility_state = Column(String)  # NORMAL/HIGH/EXTREME
    model_version = Column(String)

class HistoricalData(Base):
    __tablename__ = 'historical_data'
    id = Column(String, primary_key=True)
    symbol = Column(String)
    timeframe = Column(String)
    timestamp = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class PerformanceReport(Base):
    __tablename__ = 'performance_reports'
    id = Column(Integer, primary_key=True)
    report_date = Column(DateTime, default=datetime.utcnow)
    period = Column(String)  # DAILY, WEEKLY, MONTHLY
    total_signals = Column(Integer)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    details = Column(JSON)

# Create engine
engine = create_engine(
    Config.DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    connect_args={'sslmode': 'disable'} if Config.DATABASE_URL.startswith('postgres') else {}
)

# Test database connection
try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection successful")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    sentry_sdk.capture_exception(e)

# =====================
# DATABASE MIGRATION
# =====================
def migrate_database():
    """Handle database schema migrations"""
    inspector = inspect(engine)
    
    # Portfolio table migrations
    if 'portfolio' in inspector.get_table_names():
        portfolio_columns = [col['name'] for col in inspector.get_columns('portfolio')]
        missing_columns = []
        
        if 'total_value' not in portfolio_columns:
            missing_columns.append('total_value FLOAT DEFAULT 10000')
        
        # Add missing columns
        if missing_columns:
            logger.info("Migrating portfolio table")
            with engine.begin() as conn:
                for column_def in missing_columns:
                    try:
                        conn.execute(text(f"ALTER TABLE portfolio ADD COLUMN {column_def}"))
                        logger.info(f"Added column: {column_def}")
                    except Exception as e:
                        logger.error(f"Migration error: {e}")
    
    # Signals table migrations
    if 'signals' in inspector.get_table_names():
        signals_columns = [col['name'] for col in inspector.get_columns('signals')]
        missing_columns = []
        
        if 'model_version' not in signals_columns:
            missing_columns.append('model_version VARCHAR')
        if 'volatility_state' not in signals_columns:
            missing_columns.append('volatility_state VARCHAR')
        
        # Add missing columns
        if missing_columns:
            logger.info("Migrating signals table")
            with engine.begin() as conn:
                for column_def in missing_columns:
                    try:
                        conn.execute(text(f"ALTER TABLE signals ADD COLUMN {column_def}"))
                        logger.info(f"Added column: {column_def}")
                    except Exception as e:
                        logger.error(f"Migration error: {e}")
    
    # Create tables
    Base.metadata.create_all(engine)

# Run migrations
migrate_database()

Session = scoped_session(sessionmaker(bind=engine, autoflush=False))

# Initialize portfolio if not exists
def initialize_portfolio():
    session = Session()
    try:
        portfolio = session.query(Portfolio).first()
        if not portfolio:
            portfolio = Portfolio(total_risk=0.0, total_value=Config.PORTFOLIO_VALUE)
            session.add(portfolio)
            session.commit()
            logger.info("Created new portfolio record")
        else:
            # Initialize total_value if missing
            if portfolio.total_value is None or portfolio.total_value == 0:
                portfolio.total_value = Config.PORTFOLIO_VALUE
                session.commit()
                logger.info(f"Initialized portfolio value to ${Config.PORTFOLIO_VALUE:,.2f}")
    except Exception as e:
        logger.error(f"Portfolio initialization error: {e}")
        session.rollback()
        sentry_sdk.capture_exception(e)
    finally:
        Session.remove()

initialize_portfolio()

# =====================
# VOLATILITY SYSTEM - ENHANCED WITH CORRELATION
# =====================
class VolatilitySystem:
    def __init__(self):
        self.volatility_state = "NORMAL"
        self.last_update = datetime.utcnow()
        self.aggregate_volatility = 0.0
        self.correlation_matrix = {}
    
    def calculate_correlation(self, data: Dict[str, pd.DataFrame]) -> dict:
        """Calculate correlation matrix for major cryptocurrencies"""
        closes = {}
        for symbol, df in data.items():
            if len(df) > 100:
                closes[symbol] = df['close'].pct_change().dropna()
        
        if len(closes) < 3:
            return {}
        
        # Create correlation matrix
        corr_matrix = {}
        symbols = list(closes.keys())
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if len(closes[sym1]) == len(closes[sym2]):
                    corr, _ = spearmanr(closes[sym1], closes[sym2])
                    corr_matrix[f"{sym1.split('/')[0]}-{sym2.split('/')[0]}"] = corr
        
        return corr_matrix
    
    def update_state(self, data: Dict[str, pd.DataFrame]):
        """Update volatility state based on multiple assets and correlations"""
        if not data:
            return
            
        try:
            volatilities = []
            # Calculate volatility for top 3 coins
            top_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            for symbol in top_symbols:
                df = data.get(symbol)
                if df is not None and len(df) >= 50:                    
                    # Calculate volatility as standard deviation of returns
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 20:
                        volatility = returns.rolling(20).std().iloc[-1] * 100
                        volatilities.append(volatility)
            
            if not volatilities:
                return
                
            # Use average volatility of top coins
            self.aggregate_volatility = np.mean(volatilities)
            
            # Calculate correlations
            self.correlation_matrix = self.calculate_correlation(data)
            
            # Update volatility state
            if self.aggregate_volatility > 3.5:
                self.volatility_state = "EXTREME"
            elif self.aggregate_volatility > 2.5:
                self.volatility_state = "HIGH"
            else:
                self.volatility_state = "NORMAL"
                
            logger.info(f"Volatility State: {self.volatility_state} | Aggregate Volatility: {self.aggregate_volatility:.2f}%")
            self.last_update = datetime.utcnow()
        except Exception as e:
            logger.error(f"Volatility update error: {e}")
            sentry_sdk.capture_exception(e)

    def get_trading_rules(self):
        """Get adaptive trading rules based on volatility state"""
        rules = {
            "position_size_multiplier": 1.0,
            "max_symbols": len(Config.SYMBOLS),
            "risk_multiplier": 1.0,
            "min_volume_ratio": Config.MIN_BREAKOUT_VOLUME_RATIO,
            "leverage": min(3, Config.MAX_LEVERAGE)
        }
        
        if self.volatility_state == "HIGH":
            return {
                "position_size_multiplier": 0.7,
                "max_symbols": max(3, len(Config.SYMBOLS) // 2),
                "risk_multiplier": 0.8,
                "min_volume_ratio": Config.MIN_BREAKOUT_VOLUME_RATIO * 1.2,
                "leverage": min(2, Config.MAX_LEVERAGE)
            }
        elif self.volatility_state == "EXTREME":
            return {
                "position_size_multiplier": 0.5,
                "max_symbols": 2,
                "risk_multiplier": 0.6,
                "min_volume_ratio": Config.MIN_BREAKOUT_VOLUME_RATIO * 1.5,
                "leverage": 1
            }
        return rules

# =====================
# ML MODEL - UPGRADED TO LIGHTGBM WITH IMPROVED CONFIG
# =====================
try:
    import lightgbm as lgb
    logger.info("Using LightGBM for models")
    USE_LGB = True
except ImportError:
    logger.info("LightGBM not available, using GradientBoosting")
    USE_LGB = False

class TradingModel:
    def __init__(self, symbol: str, timeframe: str, version=Config.MODEL_VERSION):
        self.symbol = symbol
        self.timeframe = timeframe
        self.version = version
        safe_symbol = symbol.replace('/', '_')
        self.model_key = f"{safe_symbol}_{timeframe}_{version}"
        self.model_path = os.path.join(Config.MODEL_SAVE_PATH, f"{self.model_key}.pkl")
        self.scaler_path = os.path.join(Config.MODEL_SAVE_PATH, f"{self.model_key}_scaler.pkl")
        
        # Create directory if not exists
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        
        # Load model if exists
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            # Initialize new model
            if USE_LGB:
                self.model = lgb.LGBMClassifier(
                    num_leaves=31,
                    max_depth=5,
                    learning_rate=0.05,
                    n_estimators=200,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                self.model = GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=10,
                    random_state=42
                )
            self.scaler = MinMaxScaler()
            self.features = None
            self.is_trained = False
            self.feature_importances_ = {}
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True
            logger.info(f"Loaded model for {self.symbol} {self.timeframe}")
            
            # Get feature importances
            if USE_LGB:
                self.feature_importances_ = dict(zip(self.model.feature_name_, self.model.feature_importances_))
            else:
                self.feature_importances_ = dict(zip(self.model.feature_names_in_, self.model.feature_importances_))
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_trained = False
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Saved model for {self.symbol} {self.timeframe}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 50:
            return pd.DataFrame()
        
        try:
            # Make a copy with only numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[numeric_cols].copy()
            
            # Technical indicators - Enhanced set
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
            df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['vpt'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
            
            # Price action features
            df['body_size'] = (df['close'] - df['open']).abs() / (df['open'] + 1e-7)
            df['candle_ratio'] = (df['high'] - df['low']) / (df['body_size'] + 1e-7)
            df['candle_type'] = np.where(df['close'] > df['open'], 1, 
                                        np.where(df['close'] < df['open'], -1, 0))
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Momentum features
            df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
            df['stoch_osc'] = ta.momentum.StochasticOscillator(
                df['high'], df['low'], df['close']).stoch()
            
            # Volatility features
            df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
            df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_upper']
            
            # Drop rows with NaN values
            df = df.dropna()
            return df
        except Exception as e:
            logger.error(f"Feature creation error: {e}")
            sentry_sdk.capture_exception(e)
            return pd.DataFrame()
    
    def train(self, data: pd.DataFrame) -> Tuple[float, float]:
        try:
            if len(data) < Config.MIN_TRAINING_DATA:  # Use configurable threshold
                logger.warning(f"Insufficient data for training: {len(data)} rows")
                return 0.0, 0.0
                
            df = self.create_features(data)
            if df.empty or 'close' not in df.columns:
                return 0.0, 0.0
                
            # Target - predict next candle direction WITHOUT data leakage
            df['next_close'] = df['close'].shift(-1)
            df = df.dropna(subset=['next_close'])
            df['target'] = (df['next_close'] > df['close']).astype(int)
            df = df.dropna()
            
            if len(df) < 1000:
                return 0.0, 0.0
                
            X = df.drop(['target', 'next_close'], axis=1, errors='ignore')
            y = df['target']
            
            # Ensure all columns are numeric
            X = X.select_dtypes(include=[np.number])
            
            if X.empty:
                return 0.0, 0.0
                
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.features = X.columns.tolist()
            
            # Use time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)  # Reduced to 3 splits for efficiency
            test_accs = []
            test_f1s = []
            
            for train_index, test_index in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred)
                test_accs.append(test_acc)
                test_f1s.append(test_f1)
            
            # Train on full dataset after cross-validation
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Store feature importances
            if USE_LGB:
                self.feature_importances_ = dict(zip(self.model.feature_name_, self.model.feature_importances_))
            else:
                self.feature_importances_ = dict(zip(self.features, self.model.feature_importances_))
            
            avg_test_acc = np.mean(test_accs)
            avg_test_f1 = np.mean(test_f1s)
            logger.info(f"Model {self.version} trained - Avg Test Acc: {avg_test_acc:.2%}, F1: {avg_test_f1:.2%}")
            
            # Save model after training
            self.save_model()
            
            return avg_test_acc, avg_test_f1
        except Exception as e:
            logger.exception("Training error")
            sentry_sdk.capture_exception(e)
            return 0.0, 0.0
            
    def predict(self, data: pd.DataFrame) -> Tuple[float, dict]:
        """Predict probability of next candle being bullish with explanation"""
        try:
            if not self.is_trained:
                return 0.5, {}
                
            df = self.create_features(data.copy())
            if df.empty:
                return 0.5, {}
                
            # Ensure all features are present
            missing = set(self.features) - set(df.columns)
            for col in missing:
                df[col] = 0
                
            X = df[self.features]
            
            # Ensure all columns are numeric
            X = X.select_dtypes(include=[np.number])
            
            if X.empty:
                return 0.5, {}
                
            X_scaled = self.scaler.transform(X)
            probas = self.model.predict_proba(X_scaled)
            
            # Get explanation (feature contributions for last candle)
            explanation = {}
            if self.feature_importances_:
                last_candle = X.iloc[-1].to_dict()
                for feature, value in last_candle.items():
                    # Scale importance by normalized feature value
                    explanation[feature] = {
                        'value': value,
                        'importance': self.feature_importances_.get(feature, 0),
                        'contribution': self.feature_importances_.get(feature, 0) * value
                    }
            
            return probas[-1, 1], explanation  # Probability of bullish
        except Exception as e:
            logger.exception("Prediction error")
            sentry_sdk.capture_exception(e)
            return 0.5, {}

# =====================
# TRADING ENGINE - PRODUCTION GRADE (UPDATED)
# =====================
class TradingBot:
    def __init__(self):
        self.models: Dict[str, TradingModel] = {}
        self.last_trained: Dict[str, datetime] = {}
        self.portfolio_value = Config.PORTFOLIO_VALUE
        self.volatility_system = VolatilitySystem()
        self.sentiment_analyzer = None
        self.performance_stats = {
            'total_signals': 0,
            'winning_signals': 0,
            'losing_signals': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'equity_curve': []
        }
        
        # Initialize Telegram with ApplicationBuilder
        self.application = None
        try:
            self.application = Application.builder() \
                .token(Config.TELEGRAM_TOKEN) \
                .connect_timeout(30) \
                .read_timeout(30) \
                .pool_timeout(30) \
                .build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler('status', self.telegram_status))
            self.application.add_handler(CommandHandler('signals', self.telegram_active_signals))
            self.application.add_handler(CommandHandler('cancel', self.telegram_cancel))
            
            # Start polling in a separate thread with proper event loop
            def run_polling():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.application.run_polling()
            
            self.application_thread = threading.Thread(target=run_polling, daemon=True)
            self.application_thread.start()
            logger.info("Telegram command handlers initialized")
        except Exception as e:
            logger.error(f"Telegram setup failed: {e}")
            sentry_sdk.capture_exception(e)
            self.application = None
        
        # Exchange connection will be created per request
        self.scheduler = BackgroundScheduler(timezone=pytz.utc)
        self.signal_cooldown = {}  # Track cooldown periods per symbol
        self.position_warnings = {}  # Track warnings for each position
        
        # Load portfolio state from DB
        self.load_portfolio_state()
        
        # Initialize models
        for symbol in Config.SYMBOLS:
            for timeframe in Config.TIMEFRAMES:
                key = f"{symbol}-{timeframe}"
                self.models[key] = TradingModel(symbol, timeframe)
                self.last_trained[key] = datetime.utcnow() - timedelta(hours=Config.MODEL_UPDATE_INTERVAL)
        
        # Setup scheduled tasks
        self.scheduler.add_job(self.retrain_models, 'interval', hours=Config.MODEL_UPDATE_INTERVAL)
        self.scheduler.add_job(self.monitor_markets, 'interval', minutes=5)
        self.scheduler.add_job(self.heartbeat, 'interval', minutes=30)
        self.scheduler.add_job(self.update_portfolio_value, 'interval', hours=12)
        self.scheduler.add_job(self.generate_performance_report, 'interval', hours=Config.PERFORMANCE_REPORT_INTERVAL)
        
        # News monitoring
        if Config.NEWS_API_KEY:
            self.scheduler.add_job(self.monitor_news, 'interval', minutes=30)
        
        self.scheduler.start()
        
        # Pre-populate historical data
        self.pre_populate_data()
        
        # Run initial training immediately
        self.retrain_models()
        
        # Test signal generation
        self.test_signal_generation()
    
    def pre_populate_data(self):
        """Pre-populate historical data for all symbols"""
        logger.info("Pre-populating historical data")
        for symbol in Config.SYMBOLS:
            for timeframe in Config.TIMEFRAMES:
                # Fetch and save data
                df = self.fetch_historical_data(symbol, timeframe, Config.HISTORICAL_DATA_LIMIT)
                if not df.empty:
                    self.save_historical_data(df)
                    logger.info(f"Pre-populated {len(df)} rows for {symbol} {timeframe}")
    
    def get_exchange(self) -> ccxt.Exchange:
        """Create a new exchange connection"""
        exchange = ccxt.binance(Config.EXCHANGE_CONFIG)
        # Sync time
        try:
            server_time = exchange.fetch_time()
            local_time = exchange.milliseconds()
            exchange.options['timeDifference'] = local_time - server_time
        except Exception as e:
            logger.error(f"Time sync failed: {e}")
        return exchange
    
    def load_portfolio_state(self):
        """Load portfolio state from database"""
        session = Session()
        try:
            portfolio = session.query(Portfolio).first()
            if portfolio:
                self.total_risk = portfolio.total_risk
                self.portfolio_value = portfolio.total_value
                logger.info(f"Loaded portfolio state: Value=${self.portfolio_value:,.2f}, Risk={self.total_risk:.4f}")
            else:
                self.total_risk = 0.0
                self.portfolio_value = Config.PORTFOLIO_VALUE
                logger.warning("No portfolio found, initializing state")
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
            self.total_risk = 0.0
            self.portfolio_value = Config.PORTFOLIO_VALUE
        finally:
            Session.remove()
    
    def update_portfolio_state(self, risk_delta: float = 0, value_delta: float = 0):
        """Update portfolio state in memory and database"""
        self.total_risk = max(0.0, self.total_risk + risk_delta)
        self.portfolio_value = max(100, self.portfolio_value + value_delta)
        
        session = Session()
        try:
            portfolio = session.query(Portfolio).first()
            if portfolio:
                portfolio.total_risk = self.total_risk
                portfolio.total_value = self.portfolio_value
                portfolio.last_updated = datetime.utcnow()
            else:
                portfolio = Portfolio(total_risk=self.total_risk, total_value=self.portfolio_value)
                session.add(portfolio)
            session.commit()
            logger.info(f"Updated portfolio state: Value=${self.portfolio_value:,.2f}, Risk={self.total_risk:.4f}")
        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")
            session.rollback()
        finally:
            Session.remove()
    
    def test_signal_generation(self):
        """Test signal generation on startup"""
        logger.info("Running signal generation test...")
        test_results = []
        for symbol in Config.SYMBOLS[:3]:  # Test first 3 symbols
            for timeframe in Config.TIMEFRAMES:  # Test all timeframes
                df = self.fetch_historical_data(symbol, timeframe, 500)
                if not df.empty:
                    signal = self.generate_signal(symbol, timeframe, df)
                    if signal:
                        result = f"✅ TEST SIGNAL: {symbol}/{timeframe} - {signal['direction']}"
                    else:
                        result = f"❌ NO SIGNAL: {symbol}/{timeframe} - (Check debug logs)"
                    test_results.append(result)
                    logger.info(result)
                else:
                    result = f"❌ NO DATA: {symbol}/{timeframe}"
                    test_results.append(result)
                    logger.warning(result)
        
        # Send test results via Telegram
        message = "🛠️ SIGNAL GENERATION TEST RESULTS:\n" + "\n".join(test_results)
        self.send_telegram_message(message)
    
    async def async_send_telegram_message(self, message: str, parse_mode=None):
        if not self.application:
            return
            
        try:
            await self.application.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            sentry_sdk.capture_exception(e)
    
def send_telegram_message(self, message: str, parse_mode=None):
    """
    Send a Telegram message. Works with both sync (v13) and async (v20+/v21+) bot APIs.
    - If no application/bot is available, returns silently (you can change that to raise).
    """
    if not getattr(self, "application", None):
        return

    try:
        import inspect
        import asyncio

        bot = getattr(self.application, "bot", None)
        if bot is None:
            # Try to fall back to a dispatcher-style Updater (older codepaths)
            bot = getattr(self.application, "bot", None)  # keep for clarity; you can add more fallbacks

        if bot is None:
            logger.error("No telegram bot available to send message.")
            return

        # Call send_message; it may return a coroutine (async API) or a Message (sync API)
        maybe_awaitable = bot.send_message(
            chat_id=Config.TELEGRAM_CHAT_ID,
            text=message,
            parse_mode=parse_mode
        )

        # If the call returned an awaitable (async API), schedule/run it correctly
        if inspect.isawaitable(maybe_awaitable):
            # If an event loop is already running, create a task; otherwise run it to completion.
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                asyncio.create_task(maybe_awaitable)
            else:
                # This will create and manage a new event loop for this call
                asyncio.run(maybe_awaitable)
        else:
            # sync result (already executed)
            return maybe_awaitable

    except Exception as e:
        logger.error(f"Telegram send failed: {e}", exc_info=True)

def _send():
    try:
        # Handle event loop properly
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_async_send())
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")
    finally:
        loop.close()

# Run the send in a separate thread so it doesn't block
try:
    threading.Thread(target=_send).start()
except Exception as e:
    logger.error(f"Error starting Telegram send thread: {e}")

    
    def heartbeat(self):
        """Regular status updates"""
        active_signals = 0
        try:
            session = Session()
            active_signals = session.query(Signal).filter(Signal.status == 'ACTIVE').count()
            Session.remove()
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            
        status = (
            f"Bot active | Symbols: {len(Config.SYMBOLS)} | "
            f"Active Signals: {active_signals} | "
            f"Portfolio Value: ${self.portfolio_value:,.2f} | "
            f"Risk: {self.total_risk:.2%} | "
            f"Volatility: {self.volatility_system.volatility_state} "
            f"({self.volatility_system.aggregate_volatility:.2f}%)"
        )
        logger.info(status)
        self.send_telegram_message(f"❤️ Heartbeat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}\n{status}")
    
    def update_portfolio_value(self):
        """Simulate portfolio value changes (would be real in live trading)"""
        # In real trading, this would fetch actual portfolio value
        # For simulation, we'll just increase by 5% monthly
        prev_value = self.portfolio_value
        self.portfolio_value *= 1.000685  # ~5% monthly growth compounded daily
        self.update_portfolio_state(value_delta=self.portfolio_value - prev_value)
        logger.info(f"Updated portfolio value: ${self.portfolio_value:,.2f}")
    
    def retrain_models(self) -> None:
        logger.info("Starting model retraining cycle")
        
        for symbol in Config.SYMBOLS:
            for timeframe in Config.TIMEFRAMES:
                key = f"{symbol}-{timeframe}"
                logger.info(f"Processing {key}")
                
                try:
                    # Load data
                    df = self.fetch_historical_data(symbol, timeframe, Config.HISTORICAL_DATA_LIMIT)
                    if len(df) < Config.MIN_TRAINING_DATA:  # Use configurable threshold
                        logger.warning(f"Insufficient data for training {key}: {len(df)} rows")
                        continue
                    
                    # Train model
                    train_acc, test_f1 = self.models[key].train(df)
                    self.last_trained[key] = datetime.utcnow()
                    
                    # Notify if successful
                    if test_f1 > 0.55:
                        model_type = "LightGBM" if USE_LGB else "GradientBoosting"
                        message = (f"📊 Model updated for {symbol} {timeframe}\n"
                                  f"Version: {self.models[key].version}\n"
                                  f"Model: {model_type}\n"
                                  f"Test F1: {test_f1:.2%}")
                        self.send_telegram_message(message)
                    
                    logger.info(f"Model trained for {key}. Test F1: {test_f1:.2%}")
                except Exception as e:
                    logger.exception(f"Error training model for {key}")
                    sentry_sdk.capture_exception(e)
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect market regime using price action"""
        if len(df) < 50:
            return "unknown"
            
        try:
            # Calculate ADX
            if 'adx' not in df.columns:
                df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
                
            current_adx = df['adx'].iloc[-1]
            
            # Calculate Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            current_bb_width = bb_width.iloc[-1]
            
            # Detect breakout patterns
            current_high = df['high'].iloc[-1]
            prev_high = df['high'].iloc[-2]
            resistance = df['high'].rolling(20).max().iloc[-2]
            
            # Determine regime
            if current_adx > 25 and current_bb_width > 0.05:
                return "trending"
            elif current_high > resistance and current_high > prev_high * 1.01:
                return "breakout"
            elif current_bb_width < 0.03:
                return "ranging"
            return "transition"
        except Exception as e:
            logger.error(f"Market regime detection error: {e}")
            return "unknown"
    
    def detect_breakout(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Detect price breakouts using price action with detailed logging"""
        if len(df) < 50:
            return False, None
            
        try:
            # Calculate key levels
            resistance = df['high'].rolling(20).max().iloc[-2]
            support = df['low'].rolling(20).min().iloc[-2]
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            current_close = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            # Volatility rules
            rules = self.volatility_system.get_trading_rules()
            volume_requirement = avg_volume * rules['min_volume_ratio']
            
            # Log breakout conditions for debugging
            logger.debug(
                f"Breakout check: Close: {current_close:.4f} vs "
                f"Res: {resistance:.4f}/Sup: {support:.4f} | "
                f"Vol: {current_volume:.2f} (Req: {volume_requirement:.2f})"
            )
            
            # Check for bullish breakout
            if current_close > resistance and current_volume > volume_requirement:
                logger.debug(f"Bullish breakout detected: Close {current_close} > Res {resistance}")
                return True, "LONG"
            
            # Check for bearish breakout
            if current_close < support and current_volume > volume_requirement:
                logger.debug(f"Bearish breakout detected: Close {current_close} < Sup {support}")
                return True, "SHORT"
                
            return False, None
        except Exception as e:
            logger.error(f"Breakout detection error: {e}")
            sentry_sdk.capture_exception(e)
            return False, None
    
    def calculate_position_size(self, atr: float, current_price: float) -> float:
        """Calculate position size based on volatility and correlation"""
        # Get volatility rules
        rules = self.volatility_system.get_trading_rules()
        
        # Calculate base risk
        base_risk = Config.RISK_PER_TRADE * self.portfolio_value * rules['risk_multiplier']
        
        # Calculate position size
        position_size = base_risk / (atr * 1.5)
        position_size_pct = (position_size * current_price) / self.portfolio_value
        
        # Apply volatility multiplier
        position_size_pct *= rules['position_size_multiplier']
        
        # Apply leverage
        position_size_pct *= rules['leverage']
        
        # Ensure we don't exceed total portfolio risk
        max_size = Config.MAX_PORTFOLIO_RISK - self.total_risk
        return min(position_size_pct, max_size)
    
    def fetch_historical_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        if not self.within_trading_hours():
            return pd.DataFrame()
            
        logger.debug(f"Fetching data for {symbol} {timeframe}")
        exchange = self.get_exchange()
        
        # Pagination implementation to overcome Binance's 1000 candle limit
        all_ohlcv = []
        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
        since = exchange.milliseconds() - (timeframe_ms * limit)
        last_timestamp = since
        
        for attempt in range(5):
            try:
                while len(all_ohlcv) < limit:
                    # Calculate remaining candles needed
                    remaining = limit - len(all_ohlcv)
                    fetch_limit = min(remaining, 1000)  # Binance max per request
                    
                    # Fetch OHLCV data
                    ohlcv = exchange.fetch_ohlcv(
                        symbol, 
                        timeframe, 
                        since=last_timestamp, 
                        limit=fetch_limit
                    )
                    
                    if not ohlcv:
                        break
                        
                    all_ohlcv.extend(ohlcv)
                    
                    # Update last timestamp for next request
                    last_timestamp = ohlcv[-1][0] + 1
                    
                    # Break if we've reached the current time
                    if last_timestamp > exchange.milliseconds():
                        break
                
                # Create DataFrame
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                df['id'] = df['timestamp'].astype(str) + '-' + symbol + '-' + timeframe
                
                # Handle null values
                if df.isnull().values.any():
                    logger.warning(f"{symbol} has {df.isnull().sum().sum()} null values - filling")
                    df = df.ffill().bfill()
                
                logger.info(f"Fetched {len(df)} records for {symbol} {timeframe}")
                return df
            except ccxt.NetworkError as e:
                wait = 2 ** attempt
                logger.warning(f"Network error ({attempt+1}/5): {e} - waiting {wait}s")
                time.sleep(wait)
            except ccxt.ExchangeError as e:
                if 'recvWindow' in str(e):
                    # Try to resync time
                    try:
                        server_time = exchange.fetch_time()
                        local_time = exchange.milliseconds()
                        exchange.options['timeDifference'] = local_time - server_time
                    except:
                        pass
                wait = 2 ** attempt
                logger.warning(f"Exchange error ({attempt+1}/5): {e} - waiting {wait}s")
                time.sleep(wait)
            except Exception as e:
                wait = 2 ** attempt
                logger.error(f"General error ({attempt+1}/5): {e} - waiting {wait}s")
                time.sleep(wait)
                
        logger.error(f"Failed to fetch data for {symbol} {timeframe}")
        return pd.DataFrame()
    
    def within_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        now = datetime.utcnow()
        current_hour = now.hour
        
        # Handle overnight range (e.g., 22:00 to 04:00)
        if Config.TRADING_HOURS['start'] > Config.TRADING_HOURS['end']:
            return current_hour >= Config.TRADING_HOURS['start'] or current_hour < Config.TRADING_HOURS['end']
        
        return Config.TRADING_HOURS['start'] <= current_hour < Config.TRADING_HOURS['end']
    
    def save_historical_data(self, df: pd.DataFrame) -> None:
        if df.empty or not self.within_trading_hours():
            return
            
        session = Session()
        try:
            # Get existing IDs to avoid duplicates
            existing_ids = {row[0] for row in session.query(HistoricalData.id).all()}
            
            # Filter out records that already exist
            new_records = [
                r for r in df.to_dict('records')
                if r['id'] not in existing_ids
            ]
            
            if not new_records:
                logger.info("No new records to save")
                return
                
            # Insert in batches of 500
            for i in range(0, len(new_records), 500):
                batch = new_records[i:i+500]
                session.bulk_insert_mappings(HistoricalData, batch)
                session.commit()
                logger.info(f"Saved batch of {len(batch)} records to DB")
            
            logger.info(f"Total {len(new_records)} new records saved")
        except Exception as e:
            logger.error(f"Database error: {e}")
            session.rollback()
            sentry_sdk.capture_exception(e)
        finally:
            Session.remove()
    
    def load_historical_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        session = Session()
        try:
            result = session.query(HistoricalData).filter(
                HistoricalData.symbol == symbol,
                HistoricalData.timeframe == timeframe
            ).order_by(HistoricalData.timestamp.desc()).limit(limit).all()
            
            if not result:
                logger.debug(f"No historical data for {symbol} {timeframe}")
                return pd.DataFrame()
                
            data = [{
                'timestamp': r.timestamp,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume,
                'symbol': r.symbol,
                'timeframe': r.timeframe
            } for r in result]
            
            df = pd.DataFrame(data).sort_values('timestamp')
            logger.info(f"Loaded {len(df)} records from DB for {symbol} {timeframe}")
            return df
        except Exception as e:
            logger.error(f"Load data error: {e}")
            sentry_sdk.capture_exception(e)
            return self.fetch_historical_data(symbol, timeframe, limit)
        finally:
            Session.remove()
    
    def generate_signal(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[Dict]:
        if not self.within_trading_hours():
            return None
            
        key = f"{symbol}-{timeframe}"
        if len(data) < 50:
            logger.debug(f"Insufficient data for {key}: {len(data)} rows")
            return None
            
        try:
            # Update volatility state using top coins
            if symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
                # Create data for volatility system
                volatility_data = {
                    'BTC/USDT': self.load_historical_data('BTC/USDT', '1h', 100),
                    'ETH/USDT': self.load_historical_data('ETH/USDT', '1h', 100),
                    'BNB/USDT': self.load_historical_data('BNB/USDT', '1h', 100)
                }
                self.volatility_system.update_state(volatility_data)
            
            # Detect market regime
            market_regime = self.detect_market_regime(data)
            
            # Skip ranging markets except for breakouts
            if market_regime == "ranging":
                logger.debug(f"Skipping {symbol} {timeframe} - ranging market")
                return None
                
            # Detect breakout
            breakout, direction = self.detect_breakout(data)
            if not breakout:
                logger.debug(f"No breakout detected for {symbol} {timeframe}")
                return None
                
            # Get ML confirmation if enabled
            confidence = 0.7  # Base confidence for breakout
            ml_explanation = {}
            if Config.USE_ML_CONFIRMATION:
                proba, explanation = self.models[key].predict(data)
                ml_explanation = explanation
                
                # Relaxed thresholds (updated)
                if direction == "LONG" and proba < 0.55:  # Reduced from 0.6
                    logger.debug(f"ML confirmation failed for LONG: {proba:.2%}")
                    return None
                if direction == "SHORT" and proba > 0.45:  # Increased from 0.4
                    logger.debug(f"ML confirmation failed for SHORT: {proba:.2%}")
                    return None
                confidence = max(confidence, abs(proba - 0.5) * 2)
            
            # TEMPORARY DEBUG MODE OVERRIDE
            if Config.DEBUG_MODE:
                logger.warning("DEBUG MODE: Forcing signal generation")
                confidence = 0.75
                market_regime = "BREAKOUT"
            
            # Calculate ATR
            if 'atr' not in data.columns:
                data['atr'] = ta.volatility.AverageTrueRange(
                    data['high'], data['low'], data['close']
                ).average_true_range()
            atr = data['atr'].iloc[-1]
            
            current_price = float(data['close'].iloc[-1])
            
            # Calculate position size
            position_size = self.calculate_position_size(atr, current_price)
            if position_size <= 0.001:  # Minimum position size
                logger.debug(f"Position size too small: {position_size:.4%}")
                return None
                
            # Calculate risk levels
            if direction == "LONG":
                stop_loss = current_price - (atr * 1.5)
                take_profits = [
                    current_price + (atr * 1.0),
                    current_price + (atr * 2.0),
                    current_price + (atr * 3.0)
                ]
            else:  # SHORT
                stop_loss = current_price + (atr * 1.5)
                take_profits = [
                    current_price - (atr * 1.0),
                    current_price - (atr * 2.0),
                    current_price - (atr * 3.0)
                ]
            
            signal_id = f"{symbol}-{timeframe}-{int(time.time())}"
            
            logger.info(f"Generated signal for {symbol} {timeframe}: {direction} (Regime: {market_regime})")
            return {
                'id': signal_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.utcnow(),
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profits': take_profits,
                'confidence': confidence,
                'market_regime': market_regime,
                'position_size': position_size,
                'volatility_state': self.volatility_system.volatility_state,
                'model_version': self.models[key].version,
                'ml_explanation': ml_explanation,
                'status': 'ACTIVE'
            }
        except Exception as e:
            logger.exception("Signal generation error")
            sentry_sdk.capture_exception(e)
            return None
    
    def has_active_signal(self, symbol: str, timeframe: str) -> bool:
        session = Session()
        try:
            count = session.query(Signal).filter(
                Signal.symbol == symbol,
                Signal.timeframe == timeframe,
                Signal.status == 'ACTIVE'
            ).count()
            return count > 0
        except Exception as e:
            logger.error(f"Error checking active signals: {e}")
            sentry_sdk.capture_exception(e)
            return False
        finally:
            Session.remove()
    
    def in_cooldown(self, symbol: str, timeframe: str) -> bool:
        """Check cooldown period after signal closure"""
        key = f"{symbol}-{timeframe}"
        last_closed = self.signal_cooldown.get(key)
        if not last_closed:
            return False
            
        # Cooldown based on timeframe
        cooldown_minutes = {
            '15m': 60,    # 1 hour cooldown
            '1h': 120,    # 2 hours cooldown
            '4h': 240     # 4 hours cooldown
        }.get(timeframe, 60)
            
        cooldown_end = last_closed + timedelta(minutes=cooldown_minutes)
        return datetime.utcnow() < cooldown_end
    
    def monitor_markets(self) -> None:
        if not self.within_trading_hours():
            logger.info("Outside trading hours, skipping market monitoring")
            return
            
        logger.info("Starting market monitoring cycle")
        try:
            # Create data for volatility system
            volatility_data = {}
            for symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
                df = self.fetch_historical_data(symbol, '1h', 100)
                if not df.empty:
                    volatility_data[symbol] = df
            
            # Update volatility state
            if volatility_data:
                self.volatility_system.update_state(volatility_data)
            
            # Get trading rules based on volatility
            rules = self.volatility_system.get_trading_rules()
            max_symbols = rules['max_symbols']
            
            # Process each symbol in parallel
            signals_generated = 0
            symbols_to_process = Config.SYMBOLS.copy()
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_symbol = {
                    executor.submit(
                        self.process_symbol, 
                        symbol, 
                        rules, 
                        max_symbols - signals_generated
                    ): symbol 
                    for symbol in symbols_to_process
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        count = future.result()
                        signals_generated += count
                        if signals_generated >= max_symbols:
                            logger.info(f"Reached max symbols limit: {max_symbols}")
                            break
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        sentry_sdk.capture_exception(e)
                    
        except Exception as e:
            logger.exception("Market monitoring error")
            sentry_sdk.capture_exception(e)
    
    def process_symbol(self, symbol: str, rules: dict, remaining_slots: int) -> int:
        """Process a single symbol and return number of signals generated"""
        signals_generated = 0
        
        # Process each timeframe
        for timeframe in Config.TIMEFRAMES:
            if signals_generated >= remaining_slots:
                break
                
            # Check cooldown
            if self.in_cooldown(symbol, timeframe):
                continue
                
            # Check active signal
            if self.has_active_signal(symbol, timeframe):
                continue
                
            # Fetch data
            df = self.fetch_historical_data(symbol, timeframe, 500)
            if df.empty or len(df) < 50:
                continue
                
            # Generate signal
            signal = self.generate_signal(symbol, timeframe, df)
            if not signal:
                continue
                
            # Check position size
            if signal['position_size'] <= 0.001:  # Minimum position size
                continue
                
            # Save signal to database
            session = Session()
            try:
                # Create DB signal object
                db_signal = Signal(
                    id=signal['id'],
                    symbol=signal['symbol'],
                    timeframe=signal['timeframe'],
                    timestamp=signal['timestamp'],
                    direction=signal['direction'],
                    entry_price=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    take_profits=json.dumps(signal['take_profits']),
                    confidence=signal['confidence'],
                    market_regime=signal['market_regime'],
                    position_size=signal['position_size'],
                    volatility_state=signal['volatility_state'],
                    model_version=signal['model_version'],
                    status=signal['status']
                )
                session.merge(db_signal)
                session.commit()
                
                # Update total risk
                self.update_portfolio_state(risk_delta=signal['position_size'])
                
                # Format Telegram message
                position_size_usd = signal['position_size'] * self.portfolio_value
                leverage = rules['leverage']
                message = (
                    f"🚀 *NEW SIGNAL* 🚀\n"
                    f"*Symbol:* {signal['symbol']} ({signal['timeframe']})\n"
                    f"*Market:* {signal['market_regime'].capitalize()}\n"
                    f"*Volatility:* {signal['volatility_state'].capitalize()}\n"
                    f"*Direction:* {signal['direction']}\n"
                    f"*Entry:* {signal['entry_price']:.4f}\n"
                    f"*SL:* {signal['stop_loss']:.4f}\n"
                    f"*TP1:* {signal['take_profits'][0]:.4f}\n"
                    f"*TP2:* {signal['take_profits'][1]:.4f}\n"
                    f"*TP3:* {signal['take_profits'][2]:.4f}\n"
                    f"*Confidence:* {signal['confidence']:.2%}\n"
                    f"*Position Size:* ${position_size_usd:,.2f} ({signal['position_size']:.2%} of portfolio)\n"
                    f"*Leverage:* {leverage}x"
                )
                
                # Send notification
                try:
                    self.send_telegram_message(message, parse_mode="Markdown")
                    logger.info(f"Sent signal for {symbol} {signal['timeframe']}")
                    signals_generated += 1
                except Exception as e:
                    logger.error(f"Telegram send error: {e}")
            except Exception as e:
                logger.error(f"Error saving signal: {e}")
                session.rollback()
                sentry_sdk.capture_exception(e)
            finally:
                Session.remove()
            
            # Check existing signals with new data
            self.check_active_signals(symbol, df)
        
        return signals_generated
    
    def monitor_news(self):
        """Monitor cryptocurrency news for market-moving events with sentiment analysis"""
        if not Config.NEWS_API_KEY or not self.within_trading_hours():
            return
            
        logger.info("Checking cryptocurrency news")
        try:
            # Lazy-load sentiment analyzer
            if self.sentiment_analyzer is None:
                try:
                    # Use a lightweight model
                    from transformers import pipeline
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis", 
                        model="distilbert-base-uncased-finetuned-sst-2-english"
                    )
                    logger.info("Loaded sentiment analysis model")
                except Exception as e:
                    logger.error(f"Failed to load sentiment analyzer: {e}")
                    return
            
            url = f"https://newsapi.org/v2/everything?q=cryptocurrency&sources={','.join(Config.NEWS_SOURCES)}&apiKey={Config.NEWS_API_KEY}"
            response = requests.get(url)
            articles = response.json().get('articles', [])
            
            important_news = []
            for article in articles[:5]:  # Check top 5 articles
                title = article['title']
                try:
                    # Analyze sentiment
                    result = self.sentiment_analyzer(title)
                    sentiment = result[0]
                    
                    # Only consider strong negative sentiment
                    if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.9:
                        important_news.append(f"⚠️ {title} (Negative: {sentiment['score']:.0%})")
                except Exception:
                    # Fallback to keyword detection if sentiment analysis fails
                    keywords = ['hack', 'regulation', 'ban', 'crash', 'scam']
                    if any(kw in title.lower() for kw in keywords):
                        important_news.append(f"⚠️ {title}")
            
            if important_news:
                message = "📰 *IMPORTANT CRYPTO NEWS*\n" + "\n".join(important_news[:3])
                self.send_telegram_message(message, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"News monitoring error: {e}")
            sentry_sdk.capture_exception(e)
    
    def check_signal_health(self, signal: Signal, current_price: float, data: pd.DataFrame) -> Tuple[bool, str]:
        """Check if a position is at risk and should be closed early"""
        warnings = []
        
        # 1. Calculate distance to stop loss
        if signal.direction == 'LONG':
            distance_to_sl = current_price - signal.stop_loss
            entry_to_sl = signal.entry_price - signal.stop_loss
            sl_proximity = 1 - (distance_to_sl / entry_to_sl) if entry_to_sl != 0 else 1.0
        else:  # SHORT
            distance_to_sl = signal.stop_loss - current_price
            entry_to_sl = signal.stop_loss - signal.entry_price
            sl_proximity = 1 - (distance_to_sl / entry_to_sl) if entry_to_sl != 0 else 1.0
        
        # If position is >60% of the way to SL
        if sl_proximity > 0.6:
            warnings.append(f"Close to SL ({sl_proximity:.0%} of the way)")
        
        # 2. Check if volatility has increased significantly
        if 'atr' in data.columns:
            current_atr = data['atr'].iloc[-1]
            entry_atr = abs(signal.entry_price - signal.stop_loss) / 1.5
            if current_atr > entry_atr * 1.8:
                warnings.append(f"Volatility increased (ATR: {current_atr:.4f} vs entry {entry_atr:.4f})")
        
        # 3. Check if position has been open too long without progress
        time_open = datetime.utcnow() - signal.timestamp
        timeframe_sec = self.get_exchange().parse_timeframe(signal.timeframe)
        expected_duration = timedelta(seconds=timeframe_sec * 4)  # 4 candles
        
        if time_open > expected_duration:
            progress = 0
            if signal.direction == 'LONG':
                progress = (current_price - signal.entry_price) / (signal.take_profits[0] - signal.entry_price) if signal.take_profits else 0
            else:
                progress = (signal.entry_price - current_price) / (signal.entry_price - signal.take_profits[0]) if signal.take_profits else 0
                
            if progress < 0.2:  # Less than 20% progress toward first TP
                warnings.append(f"Stalled position ({time_open.seconds//60} mins, {progress:.0%} progress)")
        
        if warnings:
            return True, " | ".join(warnings)
        return False, ""
    
    def check_active_signals(self, symbol: str, df: pd.DataFrame) -> None:
        session = Session()
        try:
            # Get active signals for this symbol
            signals = session.query(Signal).filter(
                Signal.symbol == symbol,
                Signal.status == 'ACTIVE'
            ).all()
            
            if not signals:
                return
                
            current_price = float(df['close'].iloc[-1])
            
            for signal in signals:
                # Convert JSON string to list if needed
                if isinstance(signal.take_profits, str):
                    try:
                        signal.take_profits = json.loads(signal.take_profits)
                    except:
                        signal.take_profits = []
                
                # Check if position is at risk
                at_risk, risk_reason = self.check_signal_health(signal, current_price, df)
                if at_risk:
                    # Only warn once per signal every 30 minutes
                    last_warned = self.position_warnings.get(signal.id)
                    if not last_warned or (datetime.utcnow() - last_warned) > timedelta(minutes=30):
                        self.send_risk_warning(signal, current_price, risk_reason)
                        self.position_warnings[signal.id] = datetime.utcnow()
                
                # Check SL hit
                if (signal.direction == 'LONG' and current_price <= signal.stop_loss) or \
                   (signal.direction == 'SHORT' and current_price >= signal.stop_loss):
                    self.close_signal(signal.id, current_price, "SL")
                    continue
                    
                # Check TP hits
                tp_hit = None
                for i, tp in enumerate(signal.take_profits):
                    if (signal.direction == 'LONG' and current_price >= tp) or \
                       (signal.direction == 'SHORT' and current_price <= tp):
                        tp_hit = i+1
                        break
                        
                if tp_hit:
                    self.close_signal(signal.id, current_price, f"TP{tp_hit}")
        except Exception as e:
            logger.exception("Error checking active signals")
            sentry_sdk.capture_exception(e)
        finally:
            Session.remove()
    
    def send_risk_warning(self, signal: Signal, current_price: float, reason: str):
        """Send warning about a risky position"""
        # Calculate P/L
        if signal.direction == 'LONG':
            pnl = (current_price - signal.entry_price) / signal.entry_price
        else:
            pnl = (signal.entry_price - current_price) / signal.entry_price
            
        # Format message
        message = (
            f"⚠️ *POSITION AT RISK* ⚠️\n"
            f"*Symbol:* {signal.symbol} ({signal.timeframe})\n"
            f"*Direction:* {signal.direction}\n"
            f"*Entry:* {signal.entry_price:.4f}\n"
            f"*Current:* {current_price:.4f} ({pnl:+.2%})\n"
            f"*SL:* {signal.stop_loss:.4f}\n"
            f"*Reasons:* {reason}\n"
            f"*Consider closing position to limit risk*"
        )
        
        self.send_telegram_message(message, parse_mode="Markdown")

    def close_signal(self, signal_id: str, close_price: float, reason: str) -> None:
        logger.info(f"Closing signal {signal_id} ({reason})")
        session = Session()
        try:
            # Update signal in database
            signal = session.query(Signal).get(signal_id)
            if not signal:
                logger.error(f"Signal {signal_id} not found")
                return
                
            # Determine outcome
            if "TP" in reason:
                outcome = "WIN"
                self.performance_stats['winning_signals'] += 1
            elif reason == "SL":
                outcome = "LOSS"
                self.performance_stats['losing_signals'] += 1
            else:
                outcome = "EXPIRED"
                
            signal.status = 'CLOSED'
            signal.closed_at = datetime.utcnow()
            signal.closed_price = close_price
            signal.outcome = outcome
            
            session.commit()
            
            # Update cooldown tracker
            key = f"{signal.symbol}-{signal.timeframe}"
            self.signal_cooldown[key] = datetime.utcnow()
            
            # Update portfolio risk
            self.update_portfolio_state(risk_delta=-signal.position_size)
            
            # Calculate P/L
            pl_pct = self.calculate_pnl(signal, close_price)
            position_value = signal.position_size * self.portfolio_value
            pl_usd = pl_pct * position_value
            
            # Update performance stats
            self.performance_stats['total_signals'] += 1
            self.performance_stats['total_profit'] += pl_usd
            self.performance_stats['equity_curve'].append({
                'timestamp': datetime.utcnow(),
                'equity': self.portfolio_value + self.performance_stats['total_profit']
            })
            
            # Update max drawdown
            if self.performance_stats['equity_curve']:
                peak = max([point['equity'] for point in self.performance_stats['equity_curve']])
                current_equity = self.performance_stats['equity_curve'][-1]['equity']
                drawdown = (peak - current_equity) / peak
                self.performance_stats['max_drawdown'] = max(self.performance_stats['max_drawdown'], drawdown)
            
            # Format notification message
            message = (
                f"🔔 *SIGNAL CLOSED* 🔔\n"
                f"*Symbol:* {signal.symbol}\n"
                f"*Timeframe:* {signal.timeframe}\n"
                f"*Reason:* {reason}\n"
                f"*Direction:* {signal.direction}\n"
                f"*Entry:* {signal.entry_price:.4f}\n"
                f"*Exit:* {close_price:.4f}\n"
                f"*P/L:* {pl_pct:+.2%} (${pl_usd:+.2f})\n"
                f"*Outcome:* {outcome}"
            )
            
            self.send_telegram_message(message, parse_mode="Markdown")
                
        except Exception as e:
            logger.exception(f"Error closing signal {signal_id}")
            session.rollback()
            sentry_sdk.capture_exception(e)
        finally:
            Session.remove()
    
    def calculate_pnl(self, signal: Signal, close_price: float) -> float:
        try:
            if signal.direction == 'LONG':
                return (close_price - signal.entry_price) / signal.entry_price
            else:
                return (signal.entry_price - close_price) / signal.entry_price
        except Exception as e:
            logger.error(f"P/L calculation error: {e}")
            return 0.0
  
    def backtest(self, symbol: str, timeframe: str, days: int = 180) -> Dict[str, Any]:
        try:
            # Load historical data
            num_candles = int(days * 1440 / self.get_exchange().parse_timeframe(timeframe))
            df = self.load_historical_data(symbol, timeframe, num_candles)
            if len(df) < 100:
                logger.warning(f"Insufficient data for backtest: {len(df)} rows")
                return {"error": "Insufficient data"}

            # Simulate trading
            capital = 10000
            position = None
            trades = []
            
            for i in range(100, len(df)):
                data_slice = df.iloc[:i]
                signal = self.generate_signal(symbol, timeframe, data_slice)
                
                if signal and not position:
                    # Apply slippage to entry
                    if signal['direction'] == 'LONG':
                        entry_price = signal['entry_price'] * (1 + Config.SLIPPAGE_PCT)
                    else:
                        entry_price = signal['entry_price'] * (1 - Config.SLIPPAGE_PCT)
                    
                    # Enter trade
                    position = {
                        'entry_price': entry_price,
                        'stop_loss': signal['stop_loss'],
                        'take_profits': signal['take_profits'],
                        'direction': signal['direction'],
                        'entry_index': i
                    }
                    
                elif position:
                    current_price = df.iloc[i]['close']
                    
                    # Check SL
                    if (position['direction'] == 'LONG' and current_price <= position['stop_loss']) or \
                       (position['direction'] == 'SHORT' and current_price >= position['stop_loss']):
                        # Apply slippage to exit
                        exit_price = position['stop_loss'] * (1 - Config.SLIPPAGE_PCT) \
                            if position['direction'] == 'LONG' else position['stop_loss'] * (1 + Config.SLIPPAGE_PCT)
                            
                        trades.append({
                            'entry': position['entry_price'],
                            'exit': exit_price,
                            'direction': position['direction'],
                            'outcome': 'LOSS'
                        })
                        position = None
                    
                    # Check TPs
                    else:
                        for tp in position['take_profits']:
                            if (position['direction'] == 'LONG' and current_price >= tp) or \
                               (position['direction'] == 'SHORT' and current_price <= tp):
                                # Apply slippage to exit
                                exit_price = tp * (1 - Config.SLIPPAGE_PCT) \
                                    if position['direction'] == 'LONG' else tp * (1 + Config.SLIPPAGE_PCT)
                                    
                                trades.append({
                                    'entry': position['entry_price'],
                                    'exit': exit_price,
                                    'direction': position['direction'],
                                    'outcome': 'WIN'
                                })
                                position = None
                                break
            
            # Calculate performance metrics
            if not trades:
                return {"error": "No trades executed"}
            
            winning_trades = [t for t in trades if t['outcome'] == 'WIN']
            losing_trades = [t for t in trades if t['outcome'] == 'LOSS']
            win_rate = len(winning_trades) / len(trades)
            
            total_profit = 0
            for trade in trades:
                if trade['direction'] == 'LONG':
                    profit = (trade['exit'] - trade['entry']) / trade['entry']
                else:
                    profit = (trade['entry'] - trade['exit']) / trade['entry']
                total_profit += profit
            
            # Deduct fees
            total_profit -= Config.FEE_PCT * 2 * len(trades)
            
            # Calculate profit factor
            gross_profit = sum([(t['exit'] - t['entry']) for t in winning_trades])
            gross_loss = sum([(t['entry'] - t['exit']) for t in losing_trades])
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
            
            # Calculate max drawdown
            equity_curve = [capital]
            for trade in trades:
                if trade['direction'] == 'LONG':
                    profit = (trade['exit'] - trade['entry']) / trade['entry']
                else:
                    profit = (trade['entry'] - trade['exit']) / trade['entry']
                equity_curve.append(equity_curve[-1] * (1 + profit))
            
            peak = capital
            max_drawdown = 0
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return {
                "total_trades": len(trades),
                "win_rate": win_rate,
                "total_profit": total_profit,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown,
                "equity_curve": equity_curve
            }
        except Exception as e:
            logger.exception(f"Backtest error: {e}")
            return {"error": str(e)}
    
    def generate_performance_report(self):
        """Generate and save performance report"""
        try:
            session = Session()
            
            # Calculate metrics
            win_rate = self.performance_stats['winning_signals'] / max(1, self.performance_stats['total_signals'])
            profit_factor = 0  # Placeholder
            
            report = PerformanceReport(
                period="DAILY",
                total_signals=self.performance_stats['total_signals'],
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_drawdown=self.performance_stats['max_drawdown'],
                sharpe_ratio=0,  # Placeholder
                details=json.dumps(self.performance_stats)
            )
            
            session.add(report)
            session.commit()
            logger.info("Saved performance report")
            
            # Send summary via Telegram
            message = (
                f"📊 *PERFORMANCE REPORT*\n"
                f"*Total Signals:* {self.performance_stats['total_signals']}\n"
                f"*Win Rate:* {win_rate:.2%}\n"
                f"*Total Profit:* ${self.performance_stats['total_profit']:.2f}\n"
                f"*Max Drawdown:* {self.performance_stats['max_drawdown']:.2%}"
            )
            self.send_telegram_message(message, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Performance report error: {e}")
            sentry_sdk.capture_exception(e)
        finally:
            Session.remove()
    
    # =====================
    # TELEGRAM COMMAND HANDLERS
    # =====================
    async def telegram_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send portfolio status via Telegram"""
        if str(update.effective_user.id) not in Config.TELEGRAM_ALLOWED_IDS:
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        try:
            session = Session()
            portfolio = session.query(Portfolio).first()
            active_signals = session.query(Signal).filter(Signal.status == 'ACTIVE').count()
            
            message = (
                f"📊 *PORTFOLIO STATUS*\n"
                f"• Value: ${portfolio.total_value:,.2f}\n"
                f"• Risk Exposure: {portfolio.total_risk:.2%}\n"
                f"• Active Signals: {active_signals}\n"
                f"• Volatility State: {self.volatility_system.volatility_state}\n"
                f"• Last Updated: {portfolio.last_updated.strftime('%Y-%m-%d %H:%M')} UTC"
            )
            await update.message.reply_text(message, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Status command error: {e}")
            await update.message.reply_text("⚠️ Error fetching status")
        finally:
            Session.remove()
    
    async def telegram_active_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List active signals via Telegram"""
        if str(update.effective_user.id) not in Config.TELEGRAM_ALLOWED_IDS:
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        try:
            session = Session()
            signals = session.query(Signal).filter(Signal.status == 'ACTIVE').all()
            
            if not signals:
                await update.message.reply_text("ℹ️ No active signals")
                return
                
            message = ["🔔 *ACTIVE SIGNALS* 🔔"]
            for s in signals:
                # Convert JSON string to list if needed
                take_profits = s.take_profits
                if isinstance(take_profits, str):
                    try:
                        take_profits = json.loads(s.take_profits)
                    except:
                        take_profits = []
                
                message.append(
                    f"\n⚡️ *{s.symbol}* ({s.timeframe})\n"
                    f"• Direction: {s.direction}\n"
                    f"• Entry: {s.entry_price:.4f}\n"
                    f"• SL: {s.stop_loss:.4f}\n"
                    f"• TPs: {take_profits[0]:.4f}/{take_profits[1]:.4f}/{take_profits[2]:.4f}\n"
                    f"• ID: `{s.id}`"
                )
                
            await update.message.reply_text("\n".join(message), parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Active signals error: {e}")
            await update.message.reply_text("⚠️ Error fetching signals")
        finally:
            Session.remove()
    
    async def telegram_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Cancel a signal via Telegram"""
        if str(update.effective_user.id) not in Config.TELEGRAM_ALLOWED_IDS:
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        if not context.args:
            await update.message.reply_text("Usage: /cancel <signal_id>")
            return
            
        signal_id = context.args[0]
        try:
            session = Session()
            signal = session.query(Signal).get(signal_id)
            
            if not signal:
                await update.message.reply_text(f"❌ Signal {signal_id} not found")
                return
                
            if signal.status != 'ACTIVE':
                await update.message.reply_text(f"⚠️ Signal {signal_id} is not active")
                return
                
            # Cancel the signal
            signal.status = 'CANCELLED'
            signal.closed_at = datetime.utcnow()
            signal.outcome = 'CANCELLED'
            session.commit()
            
            # Update portfolio risk
            self.update_portfolio_state(risk_delta=-signal.position_size)
            
            # Send confirmation
            message = (
                f"🚫 *SIGNAL CANCELLED* 🚫\n"
                f"• Symbol: {signal.symbol}\n"
                f"• Timeframe: {signal.timeframe}\n"
                f"• ID: `{signal_id}`"
            )
            await update.message.reply_text(message, parse_mode="Markdown")
            logger.info(f"Cancelled signal {signal_id} via Telegram")
            
        except Exception as e:
            logger.error(f"Cancel command error: {e}")
            await update.message.reply_text("⚠️ Error cancelling signal")
            session.rollback()
        finally:
            Session.remove()

# =====================
# FLASK WEB DASHBOARD
# =====================
app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/signals')
def signals():
    session = Session()
    try:
        signals = session.query(Signal).order_by(Signal.timestamp.desc()).limit(50).all()
        signals_data = [{
            'symbol': s.symbol,
            'timeframe': s.timeframe,
            'timestamp': s.timestamp.isoformat(),
            'direction': s.direction,
            'entry_price': s.entry_price,
            'stop_loss': s.stop_loss,
            'take_profits': json.loads(s.take_profits) if isinstance(s.take_profits, str) else s.take_profits,
            'confidence': s.confidence,
            'status': s.status
        } for s in signals]
        return jsonify(signals_data)
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        return jsonify([])
    finally:
        Session.remove()

@app.route('/portfolio')
def portfolio():
    session = Session()
    try:
        portfolio = session.query(Portfolio).first()
        return jsonify({
            'total_value': portfolio.total_value,
            'total_risk': portfolio.total_risk,
            'last_updated': portfolio.last_updated.isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        return jsonify({})
    finally:
        Session.remove()

@app.route('/backtest', methods=['POST'])
def run_backtest():
    data = request.json
    symbol = data.get('symbol', 'BTC/USDT')
    timeframe = data.get('timeframe', '1h')
    days = data.get('days', 30)
    
    bot = TradingBot()
    results = bot.backtest(symbol, timeframe, days)
    return jsonify(results)

@app.route('/performance')
def performance():
    session = Session()
    try:
        reports = session.query(PerformanceReport).order_by(PerformanceReport.report_date.desc()).limit(10).all()
        reports_data = [{
            'report_date': r.report_date.isoformat(),
            'period': r.period,
            'total_signals': r.total_signals,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'max_drawdown': r.max_drawdown,
            'sharpe_ratio': r.sharpe_ratio
        } for r in reports]
        return jsonify(reports_data)
    except Exception as e:
        logger.error(f"Error fetching performance: {e}")
        return jsonify([])
    finally:
        Session.remove()

# =====================
# MAIN EXECUTION
# =====================
if __name__ == "__main__":
    # Start trading bot in separate thread
    bot = TradingBot()
    
    # Start Flask web server in main thread
    app.run(host='0.0.0.0', port=5000, debug=Config.DEBUG_MODE)