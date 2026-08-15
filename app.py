"""
===================================================================================
FLASK WEB SERVER - QUANTITATIVE STOCK PREDICTION & MARKET FORECASTING DASHBOARD
===================================================================================

Provides a REST API and serves an interactive web dashboard for real-time stock
price forecasting, feature engineering, machine learning model training, and strategy backtesting.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
import json

from stock_prediction import (
    StockDataFetcher,
    FeatureEngineer,
    StockPredictor,
    Evaluator
)

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    """Serve the main web dashboard page."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """REST API endpoint to run feature engineering, train ML model, and run backtest."""
    try:
        data = request.get_json() or {}
        ticker = data.get("ticker", "AAPL").upper().strip()
        model_type = data.get("model_type", "xgboost").lower().strip()
        start_date = data.get("start_date", "2016-01-01")
        end_date = data.get("end_date", "2024-01-01")
        
        # 1. Fetch historical OHLCV data
        fetcher = StockDataFetcher(ticker=ticker, start_date=start_date, end_date=end_date)
        raw_df = fetcher.fetch_data()
        
        # 2. Compute technical indicator features
        fe = FeatureEngineer(raw_df)
        clean_df = fe.add_technical_indicators()
        
        feature_cols = [
            'log_ret', 'sma_10_ratio', 'sma_50_ratio', 'ema_20_ratio',
            'macd', 'macd_signal', 'macd_hist', 'rsi_14',
            'bollinger_pct_b', 'bollinger_bandwidth', 'volatility_20',
            'volume_pct_change', 'volume_sma_ratio',
            'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_3', 'log_ret_lag_5'
        ]
        
        # 3. Train Model with strict time-series split
        predictor = StockPredictor(model_type=model_type, task="classification")
        X_train, X_test, y_train, y_test, dates_test, prices_test = predictor.prepare_time_series_split(
            clean_df, feature_cols, train_ratio=0.8
        )
        
        predictor.train(X_train, y_train)
        
        # 4. Predict on Test set
        y_pred = predictor.predict(X_test)
        y_proba = predictor.predict_proba(X_test)
        
        metrics = Evaluator.evaluate_classification(y_test, y_pred)
        backtest = Evaluator.backtest_strategy(prices_test, y_pred)
        
        # Next-day directional forecast (latest prediction)
        latest_pred = int(y_pred[-1])
        latest_prob = float(y_proba[-1]) * 100.0 if y_proba is not None else (65.0 if latest_pred == 1 else 35.0)
        next_day_direction = "UP (Bullish)" if latest_pred == 1 else "DOWN (Bearish)"
        
        # 5. Extract Feature Importances
        importances_series = predictor.get_feature_importances(feature_cols)
        feature_importances = []
        if importances_series is not None:
            for feat, val in importances_series.items():
                feature_importances.append({"feature": feat, "importance": round(float(val), 4)})
                
        # 6. Format time series data for Chart.js interactive rendering
        # Downsample or take last 300 days of test set for smooth browser display
        test_df = clean_df.iloc[-len(prices_test):].copy()
        
        dates_list = [d.strftime("%Y-%m-%d") for d in test_df.index]
        prices_list = [round(float(p), 2) for p in test_df['Price']]
        sma10_list = [round(float(p), 2) for p in test_df['sma_10']]
        sma50_list = [round(float(p), 2) for p in test_df['sma_50']]
        rsi_list = [round(float(r), 2) for r in test_df['rsi_14']]
        macd_list = [round(float(m), 4) for m in test_df['macd']]
        macd_signal_list = [round(float(s), 4) for s in test_df['macd_signal']]
        
        # Format cumulative equity curves
        cum_strat_list = [round(float(val), 4) for val in backtest['cum_strategy']]
        cum_bench_list = [round(float(val), 4) for val in backtest['cum_benchmark']]
        
        response_data = {
            "status": "success",
            "ticker": ticker,
            "model_type": model_type.upper(),
            "total_records": len(clean_df),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "metrics": {
                "directional_accuracy": round(metrics["Directional Accuracy (%)"], 2),
                "precision": round(metrics["Precision (UP)"], 4),
                "recall": round(metrics["Recall (UP)"], 4),
                "f1_score": round(metrics["F1-Score (UP)"], 4),
                "strategy_return": round(backtest["total_strategy_return (%)"], 2),
                "benchmark_return": round(backtest["total_benchmark_return (%)"], 2),
                "strategy_sharpe": round(backtest["sharpe_strategy"], 2),
                "benchmark_sharpe": round(backtest["sharpe_benchmark"], 2),
                "next_day_direction": next_day_direction,
                "confidence": round(latest_prob, 1)
            },
            "feature_importances": feature_importances,
            "chart_data": {
                "dates": dates_list,
                "prices": prices_list,
                "sma10": sma10_list,
                "sma50": sma50_list,
                "rsi": rsi_list,
                "macd": macd_list,
                "macd_signal": macd_signal_list,
                "cum_strategy": cum_strat_list,
                "cum_benchmark": cum_bench_list
            }
        }
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy", "service": "Stock Prediction API", "version": "1.0.0"})


if __name__ == "__main__":
    import sys
    print("[*] Starting Quantitative Stock Prediction Server on http://127.0.0.1:5000...", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False)



