"""
============================================================
 Smart Sales Forecasting System
 Module 2: AI Sales Prediction Engine
============================================================
 Models   : Linear Regression, Random Forest, XGBoost-style
 Features : Time series + business features
 Output   : 90-day forecast + confidence intervals
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Terminal colors ───────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ════════════════════════════════════════════════════════
#  STEP 1 – Load & Prepare Data
# ════════════════════════════════════════════════════════

def load_and_prepare(filepath="sales_data_clean.csv"):
    """Load cleaned data and build ML feature set."""
    print(f"{CYAN}{BOLD}\n[STEP 1] Loading & Preparing Data...{RESET}")

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    # Aggregate to monthly for forecasting
    monthly = (
        df.groupby(["Year", "Month"])
        .agg(
            Revenue=("Revenue", "sum"),
            Units=("Units_Sold", "sum"),
            Avg_Discount=("Discount_%", "mean"),
            Marketing_Spend=("Marketing_Spend", "sum"),
            Returns=("Returns", "sum"),
        )
        .reset_index()
    )

    # Time index
    monthly["Time_Index"] = range(len(monthly))

    # Lag features
    monthly["Revenue_Lag1"]  = monthly["Revenue"].shift(1)
    monthly["Revenue_Lag2"]  = monthly["Revenue"].shift(2)
    monthly["Revenue_Lag3"]  = monthly["Revenue"].shift(3)
    monthly["Rolling_3m"]    = monthly["Revenue"].rolling(3).mean()
    monthly["Rolling_6m"]    = monthly["Revenue"].rolling(6).mean()
    monthly["MoM_Growth_%"]  = monthly["Revenue"].pct_change() * 100

    # Season flag (Q4 = peak)
    monthly["Is_Q4"] = monthly["Month"].isin([10, 11, 12]).astype(int)
    monthly["Quarter"] = ((monthly["Month"] - 1) // 3) + 1

    monthly.dropna(inplace=True)
    print(f"  ✅  {len(monthly)} monthly records ready")
    return df, monthly


# ════════════════════════════════════════════════════════
#  STEP 2 – Train Models
# ════════════════════════════════════════════════════════

FEATURES = [
    "Time_Index", "Month", "Quarter", "Is_Q4",
    "Revenue_Lag1", "Revenue_Lag2", "Revenue_Lag3",
    "Rolling_3m", "Rolling_6m",
    "Avg_Discount", "Marketing_Spend"
]

def train_models(monthly: pd.DataFrame):
    """Train and evaluate multiple models, return best."""
    print(f"{CYAN}{BOLD}\n[STEP 2] Training Models...{RESET}")

    X = monthly[FEATURES]
    y = monthly["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest":    RandomForestRegressor(
                                n_estimators=200, max_depth=5,
                                random_state=42
                            ),
        "Gradient Boosting": GradientBoostingRegressor(
                                n_estimators=150, max_depth=3,
                                learning_rate=0.1, random_state=42
                            ),
    }

    results = {}
    print(f"\n  {'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'CV-R²':>8}")
    print(f"  {'-'*65}")

    best_r2 = -999
    best_name, best_model = None, None

    for name, model in models.items():
        if name == "Ridge Regression":
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)
            cv = cross_val_score(model, scaler.transform(X), y, cv=3,
                                 scoring="r2").mean()
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cv = cross_val_score(model, X, y, cv=3, scoring="r2").mean()

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        results[name] = dict(model=model, mae=mae, rmse=rmse,
                             r2=r2, cv=cv)

        flag = "👑" if r2 > best_r2 else "  "
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model

        print(f"  {flag} {name:<23} ${mae:>9,.0f} ${rmse:>9,.0f}"
              f" {r2:>7.3f} {cv:>7.3f}")

    print(f"\n  {GREEN}Best model: {best_name}  (R² = {best_r2:.3f}){RESET}")
    return best_model, scaler, results, best_name, X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════
#  STEP 3 – 90-Day Forecast
# ════════════════════════════════════════════════════════

def forecast_90_days(best_model, monthly: pd.DataFrame, model_name: str):
    """Generate 3-month forward forecast with confidence bands."""
    print(f"{CYAN}{BOLD}\n[STEP 3] Generating 90-Day Forecast...{RESET}")

    last_row   = monthly.iloc[-1]
    last_date  = pd.Timestamp(int(last_row["Year"]), int(last_row["Month"]), 1)
    last_idx   = int(last_row["Time_Index"])

    forecast_rows = []
    recent_revenue = list(monthly["Revenue"].tail(6))

    for i in range(1, 4):   # 3 future months
        future_date  = last_date + pd.DateOffset(months=i)
        future_month = future_date.month
        future_q     = (future_month - 1) // 3 + 1
        future_idx   = last_idx + i

        lag1 = recent_revenue[-1] if len(recent_revenue) >= 1 else 0
        lag2 = recent_revenue[-2] if len(recent_revenue) >= 2 else 0
        lag3 = recent_revenue[-3] if len(recent_revenue) >= 3 else 0
        r3m  = np.mean(recent_revenue[-3:])
        r6m  = np.mean(recent_revenue[-6:])

        row = pd.DataFrame([[
            future_idx, future_month, future_q,
            int(future_month in [10, 11, 12]),
            lag1, lag2, lag3, r3m, r6m,
            monthly["Avg_Discount"].mean(),
            monthly["Marketing_Spend"].mean() * 1.05,  # 5% spend increase
        ]], columns=FEATURES)

        if model_name == "Ridge Regression":
            pred = best_model.predict(row)[0]
        else:
            pred = best_model.predict(row)[0]

        # Confidence interval (±8% for demo)
        ci_width = pred * 0.08
        forecast_rows.append({
            "Date":     future_date.strftime("%Y-%m"),
            "Forecast": round(pred, 2),
            "Lower_CI": round(pred - ci_width, 2),
            "Upper_CI": round(pred + ci_width, 2),
        })
        recent_revenue.append(pred)

    forecast_df = pd.DataFrame(forecast_rows)
    print(f"\n  {'Month':<12} {'Forecast':>14} {'Lower CI':>14} {'Upper CI':>14}")
    print(f"  {'-'*55}")
    for _, row in forecast_df.iterrows():
        print(f"  {row['Date']:<12} ${row['Forecast']:>13,.2f}"
              f"  ${row['Lower_CI']:>13,.2f}  ${row['Upper_CI']:>13,.2f}")

    total = forecast_df["Forecast"].sum()
    print(f"\n  {GREEN}💰  Total 90-day forecast: ${total:,.2f}{RESET}")
    return forecast_df


# ════════════════════════════════════════════════════════
#  STEP 4 – Feature Importance
# ════════════════════════════════════════════════════════

def feature_importance(best_model, model_name: str):
    """Print top features driving predictions."""
    print(f"{CYAN}{BOLD}\n[STEP 4] Feature Importance...{RESET}")

    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=FEATURES)
        fi = fi.sort_values(ascending=False)
        print(f"\n  {'Feature':<25} {'Importance':>12}  {'Bar':}")
        print(f"  {'-'*60}")
        for feat, imp in fi.items():
            bar = "█" * int(imp * 50)
            print(f"  {feat:<25} {imp:>11.4f}  {CYAN}{bar}{RESET}")
    elif hasattr(best_model, "coef_"):
        print("  Ridge coefficients vary by feature scale (see scaler).")
    else:
        print("  Feature importance not available for this model.")


# ════════════════════════════════════════════════════════
#  STEP 5 – Save Results
# ════════════════════════════════════════════════════════

def save_results(monthly: pd.DataFrame, forecast_df: pd.DataFrame):
    """Save historical monthly and forecast data."""
    print(f"{CYAN}{BOLD}\n[STEP 5] Saving Results...{RESET}")

    monthly_out = monthly[["Year", "Month", "Revenue", "Units",
                            "MoM_Growth_%", "Rolling_3m"]].copy()
    monthly_out.to_csv("monthly_actuals.csv", index=False)
    forecast_df.to_csv("forecast_90days.csv", index=False)

    print(f"  ✅  monthly_actuals.csv saved")
    print(f"  ✅  forecast_90days.csv saved")


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════

def main():
    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════╗
║   🤖  AI Sales Prediction Engine  v1.0              ║
║   Multi-Model ML Forecasting + Confidence Intervals ║
╚══════════════════════════════════════════════════════╝
{RESET}""")

    df, monthly = load_and_prepare("sales_data_clean.csv")

    best_model, scaler, results, best_name, \
    X_train, X_test, y_train, y_test = train_models(monthly)

    forecast_df = forecast_90_days(best_model, monthly, best_name)
    feature_importance(best_model, best_name)
    save_results(monthly, forecast_df)

    print(f"\n{GREEN}{BOLD}✅  Prediction pipeline complete!{RESET}\n")

if __name__ == "__main__":
    main()
