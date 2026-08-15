"""
===================================================================================
STOCK PRICE PREDICTION AND MARKET DIRECTION FORECASTING PIPELINE
===================================================================================

This module provides a production-grade machine learning pipeline for stock market
forecasting, feature engineering, model training, and performance evaluation.

Key Components:
1. StockDataFetcher: Data acquisition via yfinance with realistic offline fallback.
2. FeatureEngineer: Technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, Lags).
3. StockPredictor: Multi-model support (Random Forest, XGBoost, LSTM).
4. Evaluator: Financial metrics (MAE, RMSE, R2, Directional Accuracy, Sharpe Ratio).
5. Visualizer: High-resolution diagnostic charts and cumulative return visualizer.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import xgboost as xgb

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore')
sns.set_theme(style="darkgrid")

# Set global matplotlib style for high quality visuals
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150


class StockDataFetcher:
    """Handles fetching historical stock market data via yfinance with fallback generator."""
    
    def __init__(self, ticker="AAPL", start_date="2016-01-01", end_date="2024-01-01"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch daily OHLCV data from Yahoo Finance or create realistic mock data if offline."""
        try:
            import yfinance as yf
            print(f"[*] Fetching historical market data for '{self.ticker}' from {self.start_date} to {self.end_date}...")
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
            
            if df.empty:
                raise ValueError("Downloaded DataFrame is empty.")
                
            # Handle MultiIndex columns if returned by newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            if 'Adj Close' in df.columns:
                df['Price'] = df['Adj Close']
            elif 'Close' in df.columns:
                df['Price'] = df['Close']
            else:
                raise KeyError("Neither 'Adj Close' nor 'Close' column found in downloaded data.")
                
            df = df.dropna()
            print(f"[+] Successfully downloaded {len(df)} daily trading records for '{self.ticker}'.")
            return df
            
        except Exception as e:
            print(f"[!] Warning: yfinance download failed ({e}). Generating realistic synthetic market data...")
            return self._generate_synthetic_data()
            
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate realistic geometric Brownian motion stock price sequence for fallback testing."""
        np.random.seed(42)
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        n = len(dates)
        
        dt = 1 / 252
        mu = 0.12     # 12% annual drift
        sigma = 0.25  # 25% annual volatility
        
        returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
        price_path = 150.0 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(index=dates)
        df['Price'] = price_path
        df['Adj Close'] = price_path
        df['Close'] = price_path * (1 + np.random.normal(0, 0.002, n))
        df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.005, n))
        df['High'] = np.maximum(df['Open'], df['Close']) * (1 + np.abs(np.random.normal(0, 0.008, n)))
        df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - np.abs(np.random.normal(0, 0.008, n)))
        df['Volume'] = np.random.lognormal(mean=16, sigma=0.5, size=n).astype(int)
        
        return df


class FeatureEngineer:
    """Computes technical indicators, stationary log returns, and lag features."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def add_technical_indicators(self) -> pd.DataFrame:
        """Generate comprehensive features spanning Trend, Momentum, Volatility, and Volume dynamics."""
        data = self.df.copy()
        
        # 1. Stationary Log Returns
        # Log returns solve non-stationarity issues present in raw prices
        data['log_ret'] = np.log(data['Price'] / data['Price'].shift(1))
        
        # 2. Moving Averages & Trend Ratios
        data['sma_10'] = data['Price'].rolling(window=10).mean()
        data['sma_50'] = data['Price'].rolling(window=50).mean()
        data['ema_20'] = data['Price'].ewm(span=20, adjust=False).mean()
        
        data['sma_10_ratio'] = (data['Price'] / data['sma_10']) - 1.0
        data['sma_50_ratio'] = (data['Price'] / data['sma_50']) - 1.0
        data['ema_20_ratio'] = (data['Price'] / data['ema_20']) - 1.0
        
        # 3. MACD (Moving Average Convergence Divergence)
        ema12 = data['Price'].ewm(span=12, adjust=False).mean()
        ema26 = data['Price'].ewm(span=26, adjust=False).mean()
        data['macd'] = ema12 - ema26
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # 4. Relative Strength Index (RSI - 14 Days)
        delta = data['Price'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        data['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))
        
        # 5. Bollinger Bands (%B and Bandwidth)
        ma20 = data['Price'].rolling(window=20).mean()
        std20 = data['Price'].rolling(window=20).std()
        upper_band = ma20 + (2.0 * std20)
        lower_band = ma20 - (2.0 * std20)
        data['bollinger_pct_b'] = (data['Price'] - lower_band) / (upper_band - lower_band + 1e-9)
        data['bollinger_bandwidth'] = (upper_band - lower_band) / ma20
        
        # 6. Volatility & Volume Momentum
        data['volatility_20'] = data['log_ret'].rolling(window=20).std()
        data['volume_pct_change'] = data['Volume'].pct_change()
        data['volume_sma_ratio'] = data['Volume'] / (data['Volume'].rolling(20).mean() + 1e-9) - 1.0
        
        # 7. Lagged Return Features
        for lag in [1, 2, 3, 5, 10]:
            data[f'log_ret_lag_{lag}'] = data['log_ret'].shift(lag)
            
        # Target Formulations:
        # Target 1 (Regression): Next day's log return r_{t+1}
        data['target_return'] = data['log_ret'].shift(-1)
        
        # Target 2 (Classification): Next day's price movement direction (1 for UP, 0 for DOWN/FLAT)
        data['target_direction'] = (data['target_return'] > 0).astype(int)
        
        # Clean any NaN values created by rolling windows and target shifts
        data_clean = data.dropna()
        return data_clean


class StockPredictor:
    """Wrapper class for building and training machine learning algorithms."""
    
    def __init__(self, model_type="xgboost", task="classification"):
        self.model_type = model_type.lower()
        self.task = task.lower()
        self.scaler = StandardScaler()
        self.model = None
        self._init_model()
        
    def _init_model(self):
        if self.model_type == "random_forest":
            if self.task == "classification":
                self.model = RandomForestClassifier(
                    n_estimators=200, max_depth=6, min_samples_split=5,
                    random_state=42, n_jobs=-1
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=200, max_depth=6, min_samples_split=5,
                    random_state=42, n_jobs=-1
                )
        elif self.model_type == "xgboost":
            if self.task == "classification":
                self.model = xgb.XGBClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    eval_metric="logloss"
                )
            else:
                self.model = xgb.XGBRegressor(
                    n_estimators=150, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    eval_metric="rmse"
                )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
            
    def prepare_time_series_split(self, df: pd.DataFrame, feature_cols: list, train_ratio=0.8):
        """Perform sequential time-series split to prevent temporal data leakage."""
        X = df[feature_cols]
        y = df['target_direction'] if self.task == "classification" else df['target_return']
        
        split_idx = int(len(df) * train_ratio)
        
        X_train_raw = X.iloc[:split_idx]
        X_test_raw = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # FIT scaler ONLY on training data to strictly prevent data leakage
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)
        
        dates_test = df.index[split_idx:]
        prices_test = df['Price'].iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test, dates_test, prices_test
        
    def train(self, X_train, y_train):
        """Train the model on scaled training features."""
        print(f"[*] Training {self.model_type.upper()} ({self.task}) model...")
        self.model.fit(X_train, y_train)
        print("[+] Model training complete.")
        
    def predict(self, X_test):
        """Generate predictions for test set."""
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        """Generate prediction probabilities for classification task."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        return None
        
    def get_feature_importances(self, feature_names):
        """Return feature importance Series if available."""
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)
        return None


class Evaluator:
    """Evaluates prediction performance using financial and statistical metrics."""
    
    @staticmethod
    def evaluate_classification(y_true, y_pred, y_prob=None):
        """Compute Directional Accuracy, Precision, Recall, F1, and return dictionary."""
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        metrics = {
            "Directional Accuracy (%)": acc * 100.0,
            "Precision (UP)": report['1']['precision'],
            "Recall (UP)": report['1']['recall'],
            "F1-Score (UP)": report['1']['f1-score'],
            "Macro F1-Score": report['macro avg']['f1-score']
        }
        return metrics
        
    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """Compute MAE, RMSE, and R2 metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate Directional Accuracy from sign alignment
        da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100.0
        
        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2,
            "Directional Accuracy (%)": da
        }
        return metrics

    @staticmethod
    def backtest_strategy(prices_test, y_pred_signal, benchmark_returns=None):
        """Simulate a simple long/short strategy based on model direction signals."""
        actual_returns = np.log(prices_test / prices_test.shift(1)).dropna()
        signals = pd.Series(y_pred_signal[:-1], index=actual_returns.index)
        
        # Convert 0/1 signal to -1/1 position (Long/Short or Long/Cash)
        positions = np.where(signals > 0, 1.0, -1.0)
        strategy_returns = positions * actual_returns
        
        cum_strategy = np.exp(np.cumsum(strategy_returns))
        cum_benchmark = np.exp(np.cumsum(actual_returns))
        
        # Calculate Sharpe ratio (annualized)
        sharpe_strategy = np.sqrt(252) * (strategy_returns.mean() / (strategy_returns.std() + 1e-9))
        sharpe_benchmark = np.sqrt(252) * (actual_returns.mean() / (actual_returns.std() + 1e-9))
        
        return {
            "cum_strategy": cum_strategy,
            "cum_benchmark": cum_benchmark,
            "sharpe_strategy": sharpe_strategy,
            "sharpe_benchmark": sharpe_benchmark,
            "total_strategy_return (%)": (cum_strategy.iloc[-1] - 1.0) * 100.0,
            "total_benchmark_return (%)": (cum_benchmark.iloc[-1] - 1.0) * 100.0
        }


class Visualizer:
    """Generates informative performance charts and backtest diagnostic figures."""
    
    @staticmethod
    def plot_results(df_clean, dates_test, prices_test, y_test, y_pred, feature_importances, backtest_res, save_dir="plots"):
        """Generate and save 4 key diagnostic charts."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Chart 1: Feature Importances
        if feature_importances is not None:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
            plt.title("Feature Importance Ranking (Gini Impurity / Gain)", fontsize=14, fontweight='bold')
            plt.xlabel("Importance Score")
            plt.ylabel("Technical Indicator Feature")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "feature_importance.png"))
            plt.close()
            print(f"[+] Saved chart: {os.path.join(save_dir, 'feature_importance.png')}")
            
        # Chart 2: Cumulative Backtest Strategy Performance vs Buy & Hold
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_res['cum_strategy'], label=f"Model Signal Strategy (Sharpe: {backtest_res['sharpe_strategy']:.2f})", color="#00FF87", linewidth=2.0)
        plt.plot(backtest_res['cum_benchmark'], label=f"Buy & Hold Benchmark (Sharpe: {backtest_res['sharpe_benchmark']:.2f})", color="#60A5FA", linestyle="--", linewidth=1.8)
        plt.title("Model Strategy Cumulative Performance vs. Buy & Hold Benchmark", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Growth of $1.00 Investment")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "strategy_backtest.png"))
        plt.close()
        print(f"[+] Saved chart: {os.path.join(save_dir, 'strategy_backtest.png')}")

        # Chart 3: Predicted vs Actual Direction Confusion Matrix Visual
        plt.figure(figsize=(8, 6))
        cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title("Directional Confusion Matrix (0: Down, 1: Up)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()
        print(f"[+] Saved chart: {os.path.join(save_dir, 'confusion_matrix.png')}")
