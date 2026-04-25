"""
============================================================
 Smart Sales Forecasting System
 Module 1: Data Cleaning & Preprocessing
============================================================
 Author  : Your Name
 Version : 1.0
 Purpose : Clean, validate, and enrich raw sales data
============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Color codes for terminal output ──────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner():
    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════╗
║   🚀  Smart Sales Data Cleaner  v1.0                ║
║   Automated Pipeline — Cleaning & Feature Engineering║
╚══════════════════════════════════════════════════════╝
{RESET}""")

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV with basic validation."""
    print(f"{YELLOW}[1/6] Loading data from: {filepath}{RESET}")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    print(f"      ✅  Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df

def report_issues(df: pd.DataFrame) -> dict:
    """Scan and report data quality issues."""
    print(f"\n{YELLOW}[2/6] Scanning data quality...{RESET}")
    issues = {
        "missing":    df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "neg_revenue": (df["Revenue"] < 0).sum() if "Revenue" in df.columns else 0,
        "neg_units":   (df["Units_Sold"] < 0).sum() if "Units_Sold" in df.columns else 0,
    }
    for k, v in issues.items():
        color = RED if v > 0 else GREEN
        print(f"      {color}{'⚠' if v>0 else '✓'}  {k.replace('_',' ').title()}: {v}{RESET}")
    return issues

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps."""
    print(f"\n{YELLOW}[3/6] Cleaning data...{RESET}")

    original_len = len(df)

    # 1. Standardise text columns
    text_cols = ["Region", "Salesperson", "Product", "Category",
                 "Customer_Segment", "Lead_Source"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    # 2. Remove duplicates
    df.drop_duplicates(inplace=True)

    # 3. Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # 4. Fill missing categoricals with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # 5. Remove negative revenue/units rows
    df = df[(df["Revenue"] >= 0) & (df["Units_Sold"] >= 0)]

    # 6. Recalculate Revenue for consistency
    if all(c in df.columns for c in ["Units_Sold", "Unit_Price", "Discount_%"]):
        df["Revenue_Recalc"] = (
            df["Units_Sold"] * df["Unit_Price"]
            * (1 - df["Discount_%"] / 100)
        ).round(2)

    removed = original_len - len(df)
    print(f"      ✅  Removed {removed} bad rows | Remaining: {len(df):,}")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based and business features."""
    print(f"\n{YELLOW}[4/6] Engineering features...{RESET}")

    df = df.copy()
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["Quarter"]     = df["Date"].dt.quarter
    df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"]   = df["Date"].dt.day_name()
    df["MonthName"]   = df["Date"].dt.strftime("%b")
    df["Is_Weekend"]  = df["Date"].dt.dayofweek >= 5

    # Business KPIs
    df["Profit_Margin_%"] = (
        (df["Revenue"] - df.get("Marketing_Spend", 0))
        / df["Revenue"] * 100
    ).round(2)

    df["Return_Rate_%"] = (
        df["Returns"] / df["Units_Sold"] * 100
    ).round(2)

    df["Revenue_per_Unit"] = (df["Revenue"] / df["Units_Sold"]).round(2)

    # Rolling 30-day revenue (sorted)
    df.sort_values("Date", inplace=True)
    df["Rolling_30d_Revenue"] = (
        df["Revenue"].rolling(window=5, min_periods=1).mean().round(2)
    )

    print(f"      ✅  Added 10 new feature columns")
    return df

def validate_output(df: pd.DataFrame):
    """Final validation checks."""
    print(f"\n{YELLOW}[5/6] Running final validation...{RESET}")
    checks = {
        "No nulls":        df.isnull().sum().sum() == 0,
        "No duplicates":   df.duplicated().sum() == 0,
        "Revenue positive":(df["Revenue"] >= 0).all(),
        "Date sorted":     df["Date"].is_monotonic_increasing,
    }
    for check, passed in checks.items():
        color = GREEN if passed else RED
        icon  = "✓" if passed else "✗"
        print(f"      {color}{icon}  {check}{RESET}")

def save_data(df: pd.DataFrame, out_path: str):
    """Save cleaned dataset."""
    print(f"\n{YELLOW}[6/6] Saving clean dataset...{RESET}")
    df.to_csv(out_path, index=False)
    print(f"      ✅  Saved → {out_path}")
    print(f"      📊  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

def summarise(df: pd.DataFrame):
    """Print a quick business summary."""
    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════╗
║              📈  DATASET SUMMARY                    ║
╠══════════════════════════════════════════════════════╣
║  Total Revenue   : ${df['Revenue'].sum():>14,.2f}          ║
║  Total Units     : {df['Units_Sold'].sum():>15,}          ║
║  Avg Deal Size   : ${df['Revenue'].mean():>14,.2f}          ║
║  Date Range      : {str(df['Date'].min().date())} → {str(df['Date'].max().date())}  ║
║  Regions         : {', '.join(df['Region'].unique()):>30}  ║
║  Products        : {df['Product'].nunique():>15}          ║
╚══════════════════════════════════════════════════════╝
{RESET}""")

def main():
    banner()
    INPUT_FILE  = "sales_data_raw.csv"
    OUTPUT_FILE = "sales_data_clean.csv"

    df = load_data(INPUT_FILE)
    report_issues(df)
    df = clean_data(df)
    df = engineer_features(df)
    validate_output(df)
    save_data(df, OUTPUT_FILE)
    summarise(df)

if __name__ == "__main__":
    main()
