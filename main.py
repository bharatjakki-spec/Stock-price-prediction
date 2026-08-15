"""
===================================================================================
MAIN DRIVER SCRIPT - STOCK PRICE PREDICTION & MARKET DIRECTION FORECASTING
===================================================================================

Runs the complete data acquisition, feature engineering, model training,
evaluation, and backtesting workflow for a user-specified stock ticker.
"""

import sys
import os
import pandas as pd

from stock_prediction import (
    StockDataFetcher,
    FeatureEngineer,
    StockPredictor,
    Evaluator,
    Visualizer
)


def run_pipeline(ticker="AAPL", start_date="2016-01-01", end_date="2024-01-01", model_type="xgboost"):
    print("=" * 80)
    print(f"       RUNNING STOCK PREDICTION PIPELINE FOR TICKER: {ticker}")
    print("=" * 80)
    
    # Step 1: Data Acquisition
    fetcher = StockDataFetcher(ticker=ticker, start_date=start_date, end_date=end_date)
    raw_df = fetcher.fetch_data()
    
    # Step 2: Preprocessing & Technical Feature Engineering
    print("\n[*] Preprocessing data & computing technical features...")
    fe = FeatureEngineer(raw_df)
    clean_df = fe.add_technical_indicators()
    print(f"[+] Clean dataset shape after indicators & target generation: {clean_df.shape}")
    
    # Define feature list
    feature_cols = [
        'log_ret', 'sma_10_ratio', 'sma_50_ratio', 'ema_20_ratio',
        'macd', 'macd_signal', 'macd_hist', 'rsi_14',
        'bollinger_pct_b', 'bollinger_bandwidth', 'volatility_20',
        'volume_pct_change', 'volume_sma_ratio',
        'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_3', 'log_ret_lag_5'
    ]
    
    # Step 3: Model Initialization & Time-Series Split (80% Train, 20% Test)
    predictor = StockPredictor(model_type=model_type, task="classification")
    X_train, X_test, y_train, y_test, dates_test, prices_test = predictor.prepare_time_series_split(
        clean_df, feature_cols, train_ratio=0.8
    )
    
    print(f"[*] Train set size: {len(X_train)} days | Test set size: {len(X_test)} days")
    
    # Step 4: Model Building & Training
    predictor.train(X_train, y_train)
    
    # Step 5: Prediction & Evaluation
    y_pred = predictor.predict(X_test)
    metrics = Evaluator.evaluate_classification(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print(f"   MODEL EVALUATION METRICS ({model_type.upper()})")
    print("=" * 50)
    for k, v in metrics.items():
        if "Accuracy" in k:
            print(f"  » {k:<30}: {v:.2f}%")
        else:
            print(f"  » {k:<30}: {v:.4f}")
            
    # Step 6: Financial Backtest & Strategy Simulation
    backtest_res = Evaluator.backtest_strategy(prices_test, y_pred)
    
    print("\n" + "=" * 50)
    print("   STRATEGY BACKTEST RESULTS vs BUY & HOLD BENCHMARK")
    print("=" * 50)
    print(f"  » Total Strategy Return    : {backtest_res['total_strategy_return (%)']:.2f}%")
    print(f"  » Total Benchmark Return   : {backtest_res['total_benchmark_return (%)']:.2f}%")
    print(f"  » Strategy Sharpe Ratio    : {backtest_res['sharpe_strategy']:.2f}")
    print(f"  » Benchmark Sharpe Ratio   : {backtest_res['sharpe_benchmark']:.2f}")
    print("=" * 50)
    
    # Step 7: Feature Importances & Visualization Generation
    feature_importances = predictor.get_feature_importances(feature_cols)
    print("\n[*] Top 5 Most Important Features:")
    if feature_importances is not None:
        for feat, val in feature_importances.head(5).items():
            print(f"    - {feat:<22}: {val:.4f}")
            
    print("\n[*] Generating diagnostic charts and saving plots to 'plots/' directory...")
    Visualizer.plot_results(
        clean_df, dates_test, prices_test, y_test, y_pred,
        feature_importances, backtest_res, save_dir="plots"
    )
    print("\n[+] Pipeline execution completed successfully!")


if __name__ == "__main__":
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "xgboost"
    run_pipeline(ticker=ticker_arg, model_type=model_arg)
