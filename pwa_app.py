"""
===================================================================================
  QuantPulse AI — PWA Flask Backend
===================================================================================
  A clean Flask server that powers the PWA mobile app.
  Serves the manifest, service worker, and the /api/predict endpoint.

  Run:
      python pwa_app.py

  Then open on phone:  http://<YOUR_LOCAL_IP>:5050
===================================================================================
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import os

from stock_prediction import (
    StockDataFetcher,
    FeatureEngineer,
    StockPredictor,
    Evaluator
)

app = Flask(__name__, template_folder="templates", static_folder="static")


# ── PWA required routes ────────────────────────────────────────────────────────

@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json",
                               mimetype="application/manifest+json")


@app.route("/sw.js")
def sw():
    return send_from_directory("static", "sw.js",
                               mimetype="application/javascript")


# ── Main app ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("mobile.html")


# ── Prediction API ─────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data       = request.get_json() or {}
        ticker     = data.get("ticker",     "AAPL").upper().strip()
        model_type = data.get("model_type", "xgboost").lower().strip()
        start_date = data.get("start_date", "2016-01-01")
        end_date   = data.get("end_date",   "2024-01-01")

        # 1. Fetch
        fetcher = StockDataFetcher(ticker=ticker,
                                   start_date=start_date,
                                   end_date=end_date)
        raw_df = fetcher.fetch_data()

        # 2. Features
        fe       = FeatureEngineer(raw_df)
        clean_df = fe.add_technical_indicators()

        feature_cols = [
            'log_ret','sma_10_ratio','sma_50_ratio','ema_20_ratio',
            'macd','macd_signal','macd_hist','rsi_14',
            'bollinger_pct_b','bollinger_bandwidth','volatility_20',
            'volume_pct_change','volume_sma_ratio',
            'log_ret_lag_1','log_ret_lag_2','log_ret_lag_3','log_ret_lag_5'
        ]

        # 3. Train
        predictor = StockPredictor(model_type=model_type, task="classification")
        X_train, X_test, y_train, y_test, dates_test, prices_test = \
            predictor.prepare_time_series_split(clean_df, feature_cols, train_ratio=0.8)
        predictor.train(X_train, y_train)

        # 4. Predict
        y_pred  = predictor.predict(X_test)
        y_proba = predictor.predict_proba(X_test)
        metrics = Evaluator.evaluate_classification(y_test, y_pred)
        backtest = Evaluator.backtest_strategy(prices_test, y_pred)

        # Next-day signal
        latest_pred  = int(y_pred[-1])
        latest_prob  = float(y_proba[-1]) * 100 if y_proba is not None else 60.0
        next_dir     = "UP" if latest_pred == 1 else "DOWN"

        # Feature importances
        fi_series = predictor.get_feature_importances(feature_cols)
        fi_list   = []
        if fi_series is not None:
            for feat, val in fi_series.items():
                fi_list.append({"feature": feat, "importance": round(float(val), 5)})

        # Chart data (last test period, max 400 points)
        test_df = clean_df.iloc[-len(prices_test):]
        step    = max(1, len(test_df) // 400)
        test_df = test_df.iloc[::step]

        dates_s  = [d.strftime("%Y-%m-%d") for d in test_df.index]
        prices_s = [round(float(p), 2) for p in test_df["Price"]]
        sma10_s  = [round(float(p), 2) for p in test_df["sma_10"]]
        sma50_s  = [round(float(p), 2) for p in test_df["sma_50"]]
        rsi_s    = [round(float(r), 2) for r in test_df["rsi_14"]]
        macd_s   = [round(float(m), 4) for m in test_df["macd"]]
        msig_s   = [round(float(s), 4) for s in test_df["macd_signal"]]
        bb_s     = [round(float(b), 4) for b in test_df["bollinger_pct_b"]]
        vol_s    = [round(float(v), 6) for v in test_df["volatility_20"]]

        # Equity curve (align with backtest result)
        cs = backtest["cum_strategy"]
        cb = backtest["cum_benchmark"]
        bt_step = max(1, len(cs) // 400)
        cum_strat_s = [round(float(v), 4) for v in cs.iloc[::bt_step]]
        cum_bench_s = [round(float(v), 4) for v in cb.iloc[::bt_step]]
        bt_dates_s  = [d.strftime("%Y-%m-%d") for d in cs.index[::bt_step]]

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        return jsonify({
            "status":        "success",
            "ticker":        ticker,
            "model_type":    model_type.upper(),
            "total_records": len(clean_df),
            "train_records": len(X_train),
            "test_records":  len(X_test),
            "metrics": {
                "accuracy":          round(metrics["Directional Accuracy (%)"], 2),
                "precision":         round(metrics["Precision (UP)"], 4),
                "recall":            round(metrics["Recall (UP)"], 4),
                "f1":                round(metrics["F1-Score (UP)"], 4),
                "macro_f1":          round(metrics["Macro F1-Score"], 4),
                "strategy_return":   round(backtest["total_strategy_return (%)"], 2),
                "benchmark_return":  round(backtest["total_benchmark_return (%)"], 2),
                "strategy_sharpe":   round(float(backtest["sharpe_strategy"]), 2),
                "benchmark_sharpe":  round(float(backtest["sharpe_benchmark"]), 2),
                "next_dir":          next_dir,
                "confidence":        round(latest_prob, 1),
            },
            "feature_importances": fi_list,
            "chart_data": {
                "dates":      dates_s,
                "prices":     prices_s,
                "sma10":      sma10_s,
                "sma50":      sma50_s,
                "rsi":        rsi_s,
                "macd":       macd_s,
                "macd_signal":msig_s,
                "bollinger":  bb_s,
                "volatility": vol_s,
            },
            "backtest_data": {
                "dates":      bt_dates_s,
                "strategy":   cum_strat_s,
                "benchmark":  cum_bench_s,
            },
            "confusion_matrix": cm,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "app": "QuantPulse AI PWA", "version": "2.0"})


if __name__ == "__main__":
    import socket
    # Get local IP so we can print it for mobile access
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"
    finally:
        s.close()

    print("\n" + "="*60)
    print("  QuantPulse AI  — PWA Mobile App")
    print("="*60)
    print(f"  Local:   http://127.0.0.1:5050")
    print(f"  Mobile:  http://{local_ip}:5050   ← open this on your phone")
    print("="*60)
    print("  On your phone: open the URL above in Chrome/Safari")
    print("  then tap  'Add to Home Screen'  to install as an app.\n")

    app.run(host="0.0.0.0", port=5050, debug=False)
