"""
Smart Sales Forecasting System v2.0
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime
import warnings
import io
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Smart Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background: #060b14; }
.kpi-card {
    background: linear-gradient(145deg,#0d1f35,#091929);
    border: 1px solid #1a3a5c;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    margin-bottom: 8px;
}
.kpi-val { font-size: 1.8rem; font-weight: 800; color: #60a5fa; }
.kpi-lbl { font-size: 0.7rem; color: #64748b; letter-spacing: 2px; margin-top: 5px; }
.kpi-pos { color: #34d399; font-size: 0.8rem; margin-top: 4px; }
.kpi-neg { color: #f87171; font-size: 0.8rem; margin-top: 4px; }
.banner {
    background: linear-gradient(90deg, #0a2540, #0d3060);
    border: 1px solid #2563eb;
    border-radius: 12px;
    padding: 16px 24px;
    margin: 16px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.banner-left { font-size: 1rem; font-weight: 700; color: #f1f5f9; }
.banner-right {
    font-size: 1.2rem; font-weight: 800; color: #fbbf24;
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 8px; padding: 6px 16px;
}
.banner-sub { font-size: 0.8rem; color: #34d399; margin-top: 4px; }
.insight-card {
    background: #0a1628;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    border-left: 4px solid #2563eb;
}
.insight-type   { font-size: 0.68rem; letter-spacing: 2px; color: #475569; margin-bottom: 4px; }
.insight-text   { font-size: 0.9rem; color: #e2e8f0; font-weight: 600; line-height: 1.5; }
.insight-action { font-size: 0.78rem; color: #60a5fa; margin-top: 6px; }
.sec-title {
    font-size: 1.1rem; font-weight: 800; color: #f1f5f9;
    border-left: 4px solid #2563eb;
    padding-left: 12px; margin: 22px 0 12px;
}
.chat-user {
    background: #1e3a5f; border-radius: 12px 12px 2px 12px;
    padding: 12px 16px; margin: 8px 0 8px 40px;
    color: #e2e8f0; font-size: 0.88rem;
}
.chat-bot {
    background: #0f2744; border: 1px solid #1a3a5c;
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px; margin: 8px 40px 8px 0;
    color: #cbd5e1; font-size: 0.88rem;
}
.alert-g { background:rgba(52,211,153,.1); border-left:4px solid #34d399; border-radius:6px; padding:12px 16px; margin:6px 0; }
.alert-y { background:rgba(251,191,36,.1); border-left:4px solid #fbbf24; border-radius:6px; padding:12px 16px; margin:6px 0; }
.alert-r { background:rgba(248,113,113,.1); border-left:4px solid #f87171; border-radius:6px; padding:12px 16px; margin:6px 0; }
section[data-testid="stSidebar"] { background: #07111e; border-right: 1px solid #1a2840; }
#MainMenu, footer, header { visibility: hidden; }
.stButton > button {
    background: linear-gradient(90deg,#1d4ed8,#2563eb);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 10px 22px;
}
.stTabs [data-baseweb="tab-list"] { background: #07111e; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #64748b; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: #1d4ed8 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Feature name mapping (human readable) ────────────────
FEATURE_NAMES = {
    "Time_Index":    "Time Progression",
    "Month":         "Month of Year",
    "Is_Q4":         "Q4 Season Flag",
    "Revenue_Lag1":  "Previous Month Revenue",
    "Revenue_Lag2":  "2 Months Ago Revenue",
    "Revenue_Lag3":  "3 Months Ago Revenue",
    "Rolling_3m":    "Last 3 Month Average",
    "Rolling_6m":    "Last 6 Month Average",
}
FEATS = list(FEATURE_NAMES.keys())

PLOT_BG = "#060b14"
GRID_C  = "#1e293b"
FONT_C  = "#94a3b8"


# ════════════════════════════════════════════════════════
#  DATA
# ════════════════════════════════════════════════════════

@st.cache_data
def clean_df(df):
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df.drop_duplicates(inplace=True)
    num = df.select_dtypes(include=np.number).columns
    df[num] = df[num].fillna(df[num].median())
    if "Date" in df.columns:
        df["Year"]      = df["Date"].dt.year
        df["Month"]     = df["Date"].dt.month
        df["Quarter"]   = df["Date"].dt.quarter
        df["MonthName"] = df["Date"].dt.strftime("%b")
    if "Returns" in df.columns and "Units_Sold" in df.columns:
        df["Return_Rate_%"] = (df["Returns"] / df["Units_Sold"] * 100).round(2)
    if "Marketing_Spend" in df.columns and "Revenue" in df.columns:
        df["ROI_%"] = ((df["Revenue"] - df["Marketing_Spend"]) / df["Marketing_Spend"] * 100).round(2)
    return df


@st.cache_data
def build_monthly(df):
    agg = {"Revenue": "sum", "Units_Sold": "sum"}
    for c in ["Discount_%", "Marketing_Spend", "Returns"]:
        if c in df.columns:
            agg[c] = "mean" if c == "Discount_%" else "sum"
    m = df.groupby(["Year", "Month"]).agg(agg).reset_index()
    m.sort_values(["Year", "Month"], inplace=True)
    m["Time_Index"]   = range(len(m))
    m["Revenue_Lag1"] = m["Revenue"].shift(1)
    m["Revenue_Lag2"] = m["Revenue"].shift(2)
    m["Revenue_Lag3"] = m["Revenue"].shift(3)
    m["Rolling_3m"]   = m["Revenue"].rolling(3).mean()
    m["Rolling_6m"]   = m["Revenue"].rolling(6).mean()
    m["Is_Q4"]        = m["Month"].isin([10, 11, 12]).astype(int)
    m["MoM_Growth_%"] = m["Revenue"].pct_change() * 100
    m.dropna(inplace=True)
    return m


@st.cache_data
def train_forecast(monthly):
    X, y = monthly[FEATS], monthly["Revenue"]
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=2,
        learning_rate=0.1, subsample=0.8, random_state=42
    )
    model.fit(X, y)
    r2  = r2_score(y, model.predict(X))
    mae = mean_absolute_error(y, model.predict(X))

    last    = monthly.iloc[-1]
    rec_rev = list(monthly["Revenue"].tail(6))
    rows    = []
    for i in range(1, 4):
        mo  = int(last["Month"] + i)
        if mo > 12:
            mo = mo - 12
        lag1 = rec_rev[-1]
        lag2 = rec_rev[-2]
        lag3 = rec_rev[-3]
        r3   = float(np.mean(rec_rev[-3:]))
        r6   = float(np.mean(rec_rev[-6:]))
        row  = pd.DataFrame(
            [[int(last["Time_Index"]) + i, mo, int(mo in [10, 11, 12]),
              lag1, lag2, lag3, r3, r6]],
            columns=FEATS
        )
        pred = float(model.predict(row)[0])
        rows.append({
            "Month_Num": mo,
            "Forecast":  round(pred, 2),
            "Lower":     round(pred - mae, 2),
            "Upper":     round(pred + mae, 2),
        })
        rec_rev.append(pred)
    return model, r2, mae, pd.DataFrame(rows)


# ════════════════════════════════════════════════════════
#  INSIGHT GENERATOR
# ════════════════════════════════════════════════════════

def generate_insights(df, monthly, forecast_df, r2, mae):
    insights = []

    # 1. Quarter growth
    last3  = monthly["Revenue"].tail(3).sum()
    next3  = forecast_df["Forecast"].sum()
    qg     = (next3 - last3) / last3 * 100
    direction = "increase" if qg > 0 else "decrease"
    icon   = "📈" if qg > 0 else "📉"
    color  = "green" if qg > 0 else "red"
    insights.append({
        "icon":   icon,
        "type":   "FORECAST",
        "color":  color,
        "text":   "Sales expected to " + direction + " by " + str(round(abs(qg), 1)) + "% next quarter.",
        "action": "Plan inventory for $" + f"{next3:,.0f}" + " target.",
    })

    # 2. Seasonality
    q4_avg  = monthly[monthly["Month"].isin([10, 11, 12])]["Revenue"].mean()
    non_avg = monthly[~monthly["Month"].isin([10, 11, 12])]["Revenue"].mean()
    if q4_avg > non_avg:
        boost = (q4_avg - non_avg) / non_avg * 100
        insights.append({
            "icon":   "🎄",
            "type":   "SEASONALITY",
            "color":  "blue",
            "text":   "Q4 shows " + str(round(boost, 0)) + "% stronger revenue than other quarters.",
            "action": "Double marketing budget in Oct-Dec for maximum ROI.",
        })
    else:
        insights.append({
            "icon":   "☀️",
            "type":   "SEASONALITY",
            "color":  "yellow",
            "text":   "Revenue is consistent across all quarters — no strong seasonal peak.",
            "action": "Create a Q4 promotion campaign to drive spikes.",
        })

    # 3. Top salesperson concentration
    if "Salesperson" in df.columns:
        sp_rev   = df.groupby("Salesperson")["Revenue"].sum().sort_values(ascending=False)
        top2_pct = sp_rev.head(2).sum() / sp_rev.sum() * 100
        top2     = sp_rev.head(2).index.tolist()
        top2_str = top2[0] + " & " + top2[1] if len(top2) > 1 else top2[0]
        color    = "yellow" if top2_pct > 55 else "green"
        action   = "Risk: over-reliance. Train remaining reps." if top2_pct > 55 else "Healthy. Reward top performers."
        insights.append({
            "icon":   "👥",
            "type":   "TEAM CONCENTRATION",
            "color":  color,
            "text":   "Top 2 reps (" + top2_str + ") drive " + str(round(top2_pct, 0)) + "% of total revenue.",
            "action": action,
        })

    # 4. Best product
    if "Product" in df.columns:
        prod_rev = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
        best     = str(prod_rev.index[0])
        best_pct = prod_rev.iloc[0] / prod_rev.sum() * 100
        insights.append({
            "icon":   "🏆",
            "type":   "PRODUCT OPPORTUNITY",
            "color":  "green",
            "text":   best + " leads with " + str(round(best_pct, 0)) + "% of total product revenue.",
            "action": "Increase " + best + " marketing for highest ROI.",
        })

    # 5. Return rate
    if "Returns" in df.columns:
        ret_rate = df["Returns"].sum() / df["Units_Sold"].sum() * 100
        color    = "red" if ret_rate > 5 else "green"
        icon     = "⚠️" if ret_rate > 5 else "✅"
        above    = "above" if ret_rate > 5 else "within"
        action   = "Investigate top returned products." if ret_rate > 5 else "Maintain current quality standards."
        insights.append({
            "icon":   icon,
            "type":   "RETURN RATE",
            "color":  color,
            "text":   "Return rate is " + str(round(ret_rate, 1)) + "% — " + above + " safe threshold.",
            "action": action,
        })

    # 6. MoM trend
    mom    = float(monthly["MoM_Growth_%"].tail(3).mean())
    color  = "green" if mom > 0 else "red"
    action = "Momentum is strong — maintain current strategy." if mom > 0 else "Growth slowing — review pricing."
    insights.append({
        "icon":   "📊",
        "type":   "GROWTH TREND",
        "color":  color,
        "text":   "Average monthly growth over last 3 months: " + str(round(mom, 1)) + "%.",
        "action": action,
    })

    # 7. Model confidence
    confidence = "High" if r2 > 0.85 else "Medium" if r2 > 0.70 else "Low"
    action     = "Predictions reliable — use for quarterly planning." if r2 > 0.85 else "Add more data to improve accuracy."
    insights.append({
        "icon":   "🤖",
        "type":   "AI CONFIDENCE",
        "color":  "blue",
        "text":   "Forecast confidence: " + confidence + " (R²=" + str(round(r2, 3)) + ", error ±$" + f"{mae:,.0f}" + ").",
        "action": action,
    })

    return insights


# ════════════════════════════════════════════════════════
#  CHATBOT
# ════════════════════════════════════════════════════════

def answer_question(question, df, monthly, forecast_df, r2, mae):
    q          = question.lower().strip()
    total_rev  = df["Revenue"].sum()
    next_fc    = forecast_df["Forecast"].iloc[0]
    total_3m   = forecast_df["Forecast"].sum()
    mom        = monthly["MoM_Growth_%"].iloc[-1] if len(monthly) > 1 else 0

    if any(w in q for w in ["total revenue", "how much revenue", "revenue total"]):
        return "Total Revenue: $" + f"{total_rev:,.2f}" + " across " + str(len(df)) + " transactions."

    if any(w in q for w in ["best month", "highest month", "top month"]):
        best = monthly.loc[monthly["Revenue"].idxmax()]
        return "Best Month: " + str(int(best["Month"])) + "/" + str(int(best["Year"])) + " — $" + f"{best['Revenue']:,.2f}"

    if any(w in q for w in ["best product", "top product"]):
        if "Product" in df.columns:
            bp  = df.groupby("Product")["Revenue"].sum().idxmax()
            val = df.groupby("Product")["Revenue"].sum().max()
            return "Best Product: " + str(bp) + " — $" + f"{val:,.2f}"

    if any(w in q for w in ["best region", "top region"]):
        if "Region" in df.columns:
            br  = df.groupby("Region")["Revenue"].sum().idxmax()
            val = df.groupby("Region")["Revenue"].sum().max()
            return "Best Region: " + str(br) + " — $" + f"{val:,.2f}"

    if any(w in q for w in ["best salesperson", "top rep", "top sales"]):
        if "Salesperson" in df.columns:
            bs  = df.groupby("Salesperson")["Revenue"].sum().idxmax()
            val = df.groupby("Salesperson")["Revenue"].sum().max()
            return "Top Salesperson: " + str(bs) + " — $" + f"{val:,.2f}"

    if any(w in q for w in ["forecast", "predict", "next month"]):
        return ("AI Forecast:\n"
                + "Next month: $" + f"{next_fc:,.2f}" + "\n"
                + "3-month total: $" + f"{total_3m:,.2f}" + "\n"
                + "Expected error: +-$" + f"{mae:,.0f}" + "\n"
                + "Model accuracy: R2=" + str(round(r2, 3)))

    if "growth" in q or "trend" in q:
        direction = "Positive momentum!" if mom > 0 else "Needs attention."
        return "MoM growth: " + str(round(mom, 1)) + "%. " + direction

    if "return" in q and "Returns" in df.columns:
        rr  = df["Returns"].sum() / df["Units_Sold"].sum() * 100
        msg = "Above 5% threshold." if rr > 5 else "Healthy."
        return "Return Rate: " + str(round(rr, 1)) + "% — " + msg

    if "discount" in q and "Discount_%" in df.columns:
        ad  = df["Discount_%"].mean()
        msg = "High — may erode margins." if ad > 8 else "Healthy level."
        return "Avg Discount: " + str(round(ad, 1)) + "% — " + msg

    if any(w in q for w in ["summary", "overview", "insights"]):
        bp = str(df.groupby("Product")["Revenue"].sum().idxmax()) if "Product" in df.columns else "N/A"
        br = str(df.groupby("Region")["Revenue"].sum().idxmax())  if "Region"  in df.columns else "N/A"
        return ("Summary:\n"
                + "Revenue: $" + f"{total_rev:,.2f}" + "\n"
                + "Transactions: " + str(len(df)) + "\n"
                + "Best Product: " + bp + "\n"
                + "Best Region: " + br + "\n"
                + "MoM Growth: " + str(round(mom, 1)) + "%\n"
                + "Next Month: $" + f"{next_fc:,.2f}" + "\n"
                + "Forecast Error: +-$" + f"{mae:,.0f}")

    return ("Try asking:\n"
            + "What is the total revenue?\n"
            + "Which product is best?\n"
            + "Who is the top salesperson?\n"
            + "What is the forecast?\n"
            + "Give me a summary")


# ════════════════════════════════════════════════════════
#  PDF REPORT
# ════════════════════════════════════════════════════════

def generate_pdf(df, monthly, forecast_df, r2, mae):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                        Spacer, Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER

        buf  = io.BytesIO()
        doc  = SimpleDocTemplate(buf, pagesize=A4,
                                 leftMargin=2*cm, rightMargin=2*cm,
                                 topMargin=2*cm,  bottomMargin=2*cm)
        BLUE = colors.HexColor("#1d4ed8")
        GREY = colors.HexColor("#64748b")

        title_s = ParagraphStyle("t", fontSize=20, fontName="Helvetica-Bold",
                                 textColor=BLUE,  spaceAfter=4, alignment=TA_CENTER)
        sub_s   = ParagraphStyle("s", fontSize=9,  fontName="Helvetica",
                                 textColor=GREY,  spaceAfter=14, alignment=TA_CENTER)
        h2_s    = ParagraphStyle("h", fontSize=12, fontName="Helvetica-Bold",
                                 textColor=colors.HexColor("#0f172a"),
                                 spaceBefore=12, spaceAfter=6)
        body_s  = ParagraphStyle("b", fontSize=9,  fontName="Helvetica",
                                 textColor=colors.HexColor("#334155"), spaceAfter=4)

        def make_table(data, widths):
            t = Table(data, colWidths=widths)
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1),
                 [colors.HexColor("#f8fafc"), colors.white]),
                ("GRID",  (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            return t

        elems = [Spacer(1, 0.5*cm)]
        elems.append(Paragraph("Smart Sales Forecasting Report", title_s))
        elems.append(Paragraph(
            "Generated: " + datetime.now().strftime("%B %d, %Y  %H:%M"),
            sub_s))
        elems.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=14))

        total_rev = df["Revenue"].sum()
        mom       = monthly["MoM_Growth_%"].iloc[-1] if len(monthly) > 1 else 0
        best_prod = str(df.groupby("Product")["Revenue"].sum().idxmax()) if "Product" in df.columns else "N/A"
        best_reg  = str(df.groupby("Region")["Revenue"].sum().idxmax())  if "Region"  in df.columns else "N/A"

        elems.append(Paragraph("Executive Summary", h2_s))
        elems.append(make_table([
            ["Metric",       "Value",                    "Metric",         "Value"],
            ["Total Revenue","$" + f"{total_rev:,.0f}",  "Transactions",   str(len(df))],
            ["Avg Deal Size","$" + f"{df['Revenue'].mean():,.0f}", "MoM Growth", str(round(mom, 1)) + "%"],
            ["Best Product", best_prod,                  "Best Region",    best_reg],
            ["Model R2",     str(round(r2, 3)),           "Forecast Error", "+-$" + f"{mae:,.0f}"],
        ], [3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm]))

        mn = ["", "Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
        elems.append(Spacer(1, 0.4*cm))
        elems.append(Paragraph("AI 90-Day Forecast", h2_s))
        fc_data = [["Month", "Forecast", "Lower CI", "Upper CI"]]
        for _, row in forecast_df.iterrows():
            fc_data.append([
                mn[int(row["Month_Num"])],
                "$" + f"{row['Forecast']:,.0f}",
                "$" + f"{row['Lower']:,.0f}",
                "$" + f"{row['Upper']:,.0f}",
            ])
        elems.append(make_table(fc_data, [3*cm, 4*cm, 4*cm, 4*cm]))

        elems.append(Spacer(1, 0.4*cm))
        elems.append(Paragraph("Auto-Generated Business Insights", h2_s))
        insights = generate_insights(df, monthly, forecast_df, r2, mae)
        for ins in insights:
            elems.append(Paragraph(ins["icon"] + " " + ins["text"], body_s))
            elems.append(Paragraph("   " + ins["action"], body_s))

        elems.append(Spacer(1, 1*cm))
        elems.append(HRFlowable(width="100%", thickness=1, color=GREY))
        elems.append(Paragraph(
            "Smart Sales Forecasting System  |  AI-Powered  |  Portfolio Project",
            ParagraphStyle("f", fontSize=7, textColor=GREY,
                           alignment=TA_CENTER, spaceBefore=6)))
        doc.build(elems)
        return buf.getvalue()

    except ImportError:
        lines = [
            "=" * 55,
            "  SMART SALES FORECASTING REPORT",
            "  " + datetime.now().strftime("%B %d, %Y"),
            "=" * 55,
            "",
            "Total Revenue : $" + f"{df['Revenue'].sum():,.2f}",
            "Model R2      : " + str(round(r2, 3)),
            "Forecast Error: +-$" + f"{mae:,.0f}",
            "",
            "90-DAY FORECAST",
            "-" * 40,
        ]
        for _, row in forecast_df.iterrows():
            lines.append("Month " + str(int(row["Month_Num"])).zfill(2)
                         + ": $" + f"{row['Forecast']:,.0f}"
                         + "  (+-$" + f"{mae:,.0f}" + ")")
        return "\n".join(lines).encode()


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════

def main():

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 10px;">
          <div style="font-size:2.5rem">📈</div>
          <div style="font-size:1rem;font-weight:800;color:#60a5fa;">SALES AI</div>
          <div style="font-size:0.68rem;color:#475569;letter-spacing:3px;">FORECASTING v2.0</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.checkbox("Use demo dataset", value=(uploaded is None))
        st.markdown("---")

    # Load data
    raw     = pd.read_csv(uploaded if uploaded else "sales_data_raw.csv")
    df      = clean_df(raw)
    monthly = build_monthly(df)
    model, r2, mae, forecast_df = train_forecast(monthly)

    with st.sidebar:
        if "Region" in df.columns:
            regions  = ["All"] + sorted(df["Region"].unique().tolist())
            sel_reg  = st.selectbox("Region",  regions)
        else:
            sel_reg = "All"
        if "Product" in df.columns:
            products = ["All"] + sorted(df["Product"].unique().tolist())
            sel_prod = st.selectbox("Product", products)
        else:
            sel_prod = "All"
        st.markdown("---")
        st.markdown("*Model Info*")
        st.caption("Algorithm : Gradient Boosting")
        st.caption("R2 Score  : " + str(round(r2, 4)))
        st.caption("MAE       : $" + f"{mae:,.0f}")

    dff = df.copy()
    if sel_reg  != "All":
        dff = dff[dff["Region"]  == sel_reg]
    if sel_prod != "All":
        dff = dff[dff["Product"] == sel_prod]

    # Header
    st.markdown("""
    <div style="background:linear-gradient(135deg,#061628,#0a1f35);
                border:1px solid #1a3a5c;border-radius:16px;
                padding:24px 28px;margin-bottom:14px;">
      <div style="font-size:1.8rem;font-weight:800;color:#60a5fa;">
        🚀 Smart Sales Forecasting System
      </div>
      <div style="color:#64748b;font-size:0.8rem;margin-top:5px;letter-spacing:1px;">
        AI-POWERED  ·  WHAT-IF SIMULATOR  ·  CHATBOT  ·  AUTO INSIGHTS  ·  PDF REPORTS
      </div>
    </div>""", unsafe_allow_html=True)

    # KPI Row
    mom = monthly["MoM_Growth_%"].iloc[-1] if len(monthly) > 1 else 0.0

    total_rev_str  = "$" + f"{dff['Revenue'].sum():,.0f}"
    total_units_str= f"{dff['Units_Sold'].sum():,}"
    avg_deal_str   = "$" + f"{dff['Revenue'].mean():,.0f}"
    trans_str      = f"{len(dff):,}"
    forecast_str   = "$" + f"{forecast_df['Forecast'].iloc[0]:,.0f}"
    mom_str        = str(round(mom, 1)) + "% MoM"

    kpis = [
        ("TOTAL REVENUE",  total_rev_str,  mom_str,              mom >= 0),
        ("TOTAL UNITS",    total_units_str,"Units sold",          True),
        ("AVG DEAL SIZE",  avg_deal_str,   "Per transaction",     True),
        ("TRANSACTIONS",   trans_str,      "Total deals",         True),
        ("NEXT FORECAST",  forecast_str,   "AI next month pred",  True),
    ]
    cols = st.columns(5)
    for col, (lbl, val, delta, pos) in zip(cols, kpis):
        delta_class = "kpi-pos" if pos else "kpi-neg"
        col.markdown(
            '<div class="kpi-card">'
            + '<div class="kpi-val">' + val + "</div>"
            + '<div class="kpi-lbl">' + lbl + "</div>"
            + '<div class="' + delta_class + '">' + delta + "</div>"
            + "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FEATURE 1: KILLER BANNER ──────────────────────────
    next_fc    = float(forecast_df["Forecast"].iloc[0])
    confidence = "High Confidence" if r2 > 0.85 else "Medium Confidence"
    mae_str    = "$" + f"{mae:,.0f}"
    r2_str     = str(round(r2, 3))

    st.markdown(
        '<div class="banner">'
        + '<div>'
        + '<div class="banner-left">🎯 Next Month AI Prediction: '
        + "<b>$" + f"{next_fc:,.0f}" + "</b></div>"
        + '<div class="banner-sub">Model Accuracy R² = ' + r2_str
        + " · " + confidence + " · Gradient Boosting</div>"
        + "</div>"
        + '<div class="banner-right">Expected Forecast Error: ±' + mae_str + "</div>"
        + "</div>",
        unsafe_allow_html=True
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard",
        "💡 Auto Insights",
        "🤖 AI Chatbot",
        "🎛️ What-If",
        "📄 PDF Report",
        "🔬 Model",
    ])

    # ════ TAB 1: DASHBOARD ════
    with tab1:
        c1, c2 = st.columns([3, 2])

        with c1:
            st.markdown('<div class="sec-title">📈 Monthly Revenue Trend</div>',
                        unsafe_allow_html=True)
            x_labels = monthly.apply(
                lambda r: str(int(r.Year)) + "-" + str(int(r.Month)).zfill(2), axis=1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_labels, y=monthly["Revenue"],
                mode="lines+markers", name="Revenue",
                line=dict(color="#60a5fa", width=2.5),
                fill="tozeroy", fillcolor="rgba(96,165,250,0.08)",
            ))
            fig.update_layout(
                paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                font=dict(color=FONT_C),
                xaxis=dict(gridcolor=GRID_C, tickangle=-45),
                yaxis=dict(gridcolor=GRID_C, tickprefix="$"),
                height=280, margin=dict(t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="sec-title">🗺️ Revenue by Region</div>',
                        unsafe_allow_html=True)
            if "Region" in dff.columns:
                reg  = dff.groupby("Region")["Revenue"].sum().reset_index()
                fig2 = px.pie(reg, values="Revenue", names="Region",
                              color_discrete_sequence=["#60a5fa","#34d399","#fbbf24","#f87171"],
                              hole=0.45)
                fig2.update_layout(
                    paper_bgcolor=PLOT_BG, font=dict(color=FONT_C),
                    height=280, margin=dict(t=10, b=0))
                st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="sec-title">📦 Product Performance</div>',
                        unsafe_allow_html=True)
            if "Product" in dff.columns:
                prod = dff.groupby("Product")["Revenue"].sum().reset_index().sort_values("Revenue")
                fig3 = px.bar(prod, x="Revenue", y="Product", orientation="h",
                              color="Revenue", color_continuous_scale="Blues")
                fig3.update_layout(
                    paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                    font=dict(color=FONT_C), height=260,
                    xaxis=dict(tickprefix="$", gridcolor=GRID_C),
                    margin=dict(t=10))
                st.plotly_chart(fig3, use_container_width=True)

        with c4:
            st.markdown('<div class="sec-title">👤 Sales Team</div>',
                        unsafe_allow_html=True)
            if "Salesperson" in dff.columns:
                sp = dff.groupby("Salesperson")["Revenue"].sum().reset_index()
                sp = sp.sort_values("Revenue", ascending=False)
                fig4 = px.bar(sp, x="Salesperson", y="Revenue",
                              color="Revenue", color_continuous_scale="Teal")
                fig4.update_layout(
                    paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                    font=dict(color=FONT_C), height=260,
                    yaxis=dict(tickprefix="$", gridcolor=GRID_C),
                    margin=dict(t=10))
                st.plotly_chart(fig4, use_container_width=True)

        # Forecast chart
        st.markdown('<div class="sec-title">🔮 AI Revenue Forecast</div>',
                    unsafe_allow_html=True)
        months_hist = monthly.apply(
            lambda r: str(int(r.Year)) + "-" + str(int(r.Month)).zfill(2), axis=1
        ).tolist()
        fc_labels = ["2024-" + str(int(r.Month_Num)).zfill(2) for _, r in forecast_df.iterrows()]

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=months_hist, y=monthly["Revenue"],
            mode="lines+markers", name="Historical",
            line=dict(color="#60a5fa", width=2.5)))
        fig5.add_trace(go.Scatter(
            x=fc_labels, y=forecast_df["Forecast"],
            mode="lines+markers", name="AI Forecast",
            line=dict(color="#34d399", width=2.5, dash="dash"),
            marker=dict(size=10, symbol="diamond")))
        fig5.add_trace(go.Scatter(
            x=fc_labels + fc_labels[::-1],
            y=list(forecast_df["Upper"]) + list(forecast_df["Lower"][::-1]),
            fill="toself", fillcolor="rgba(52,211,153,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="+-$" + f"{mae:,.0f}" + " Error Band"))
        fig5.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_C), height=360,
            xaxis=dict(gridcolor=GRID_C, tickangle=-45),
            yaxis=dict(gridcolor=GRID_C, tickprefix="$"),
            legend=dict(bgcolor="#0d1f35", bordercolor="#1a3a5c"))
        st.plotly_chart(fig5, use_container_width=True)

        # Alerts
        st.markdown('<div class="sec-title">🔔 AI Alerts</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown(
                '<div class="alert-g">✅ <b>Revenue Trend</b><br>MoM growth: '
                + str(round(mom, 1)) + "%. Positive trajectory.</div>",
                unsafe_allow_html=True)
        with a2:
            disc = dff["Discount_%"].mean() if "Discount_%" in dff.columns else 0
            cls  = "alert-y" if disc > 8 else "alert-g"
            msg  = "Consider reducing to protect margins." if disc > 8 else "Healthy level."
            st.markdown(
                '<div class="' + cls + '">🏷️ <b>Avg Discount: '
                + str(round(disc, 1)) + "%</b><br>" + msg + "</div>",
                unsafe_allow_html=True)
        with a3:
            if "Returns" in dff.columns:
                rr  = dff["Returns"].sum() / dff["Units_Sold"].sum() * 100
                cls = "alert-r" if rr > 5 else "alert-g"
                msg = "Investigate returns." if rr > 5 else "Healthy return rate."
                st.markdown(
                    '<div class="' + cls + '">📦 <b>Return Rate: '
                    + str(round(rr, 1)) + "%</b><br>" + msg + "</div>",
                    unsafe_allow_html=True)

    # ════ TAB 2: AUTO INSIGHTS ════
    with tab2:
        st.markdown('<div class="sec-title">💡 Auto Business Insight Generator</div>',
                    unsafe_allow_html=True)
        st.markdown("AI insights from your data — *clear, short, and actionable.*")

        color_border = {
            "green":  "#34d399",
            "yellow": "#fbbf24",
            "red":    "#f87171",
            "blue":   "#60a5fa",
            "purple": "#a78bfa",
        }

        insights = generate_insights(dff, monthly, forecast_df, r2, mae)
        left, right = st.columns(2)
        for i, ins in enumerate(insights):
            col = left if i % 2 == 0 else right
            border = color_border.get(ins["color"], "#60a5fa")
            with col:
                st.markdown(
                    '<div class="insight-card" style="border-left-color:' + border + ';">'
                    + '<div style="font-size:1.4rem">' + ins["icon"] + "</div>"
                    + '<div class="insight-type">' + ins["type"] + "</div>"
                    + '<div class="insight-text">' + ins["text"] + "</div>"
                    + '<div class="insight-action">→ ' + ins["action"] + "</div>"
                    + "</div>",
                    unsafe_allow_html=True)

        # Summary box
        q_growth = (forecast_df["Forecast"].sum() - monthly["Revenue"].tail(3).sum()) / \
                    monthly["Revenue"].tail(3).sum() * 100
        best_p = str(dff.groupby("Product")["Revenue"].sum().idxmax()) if "Product" in dff.columns else "N/A"
        st.markdown(
            '<div style="background:linear-gradient(135deg,#0a2540,#061628);'
            + 'border:1px solid #2563eb;border-radius:12px;padding:20px 24px;margin-top:10px;">'
            + '<div style="font-size:0.68rem;letter-spacing:3px;color:#475569;margin-bottom:10px;">'
            + "AI EXECUTIVE SUMMARY</div>"
            + '<div style="font-size:0.95rem;color:#e2e8f0;line-height:1.9;">'
            + "Revenue forecast: <b style='color:#60a5fa'>$" + f"{forecast_df['Forecast'].sum():,.0f}"
            + "</b> over next 3 months (" + str(round(q_growth, 1)) + "% vs last quarter)<br>"
            + "Model predicts with <b style='color:#34d399'>±$" + f"{mae:,.0f}"
            + "</b> expected error (R²=" + str(round(r2, 3)) + ")<br>"
            + "Top opportunity: <b style='color:#fbbf24'>" + best_p + "</b> product line"
            + "</div></div>",
            unsafe_allow_html=True)

    # ════ TAB 3: CHATBOT ════
    with tab3:
        st.markdown('<div class="sec-title">🤖 Ask Your Data Anything</div>',
                    unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "bot",
                 "text": "Hi! I am your Sales AI. Try: Give me a summary, or What is the forecast?"}
            ]

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    '<div class="chat-user">You: ' + msg["text"] + "</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="chat-bot">AI: ' + msg["text"] + "</div>",
                    unsafe_allow_html=True)

        ci, cb, cc = st.columns([5, 1, 1])
        with ci:
            user_q = st.text_input("", placeholder="Ask about your sales data...",
                                   label_visibility="collapsed", key="chat_input")
        with cb:
            if st.button("Send"):
                if user_q.strip():
                    ans = answer_question(user_q, dff, monthly, forecast_df, r2, mae)
                    st.session_state.chat_history.append({"role": "user", "text": user_q})
                    st.session_state.chat_history.append({"role": "bot",  "text": ans})
                    st.rerun()
        with cc:
            if st.button("Clear"):
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("*Quick Questions:*")
        qcols = st.columns(4)
        quick = ["Total revenue?", "Best product?", "Forecast?", "Summary"]
        for qcol, qq in zip(qcols, quick):
            with qcol:
                if st.button(qq, key="q_" + qq):
                    ans = answer_question(qq, dff, monthly, forecast_df, r2, mae)
                    st.session_state.chat_history.append({"role": "user", "text": qq})
                    st.session_state.chat_history.append({"role": "bot",  "text": ans})
                    st.rerun()

    # ════ TAB 4: WHAT-IF ════
    with tab4:
        st.markdown('<div class="sec-title">🎛️ What-If Revenue Simulator</div>',
                    unsafe_allow_html=True)
        base = float(forecast_df["Forecast"].iloc[0])

        s1, s2 = st.columns(2)
        with s1:
            disc  = st.slider("Discount Change (%)",       -10, 10,  0)
            units = st.slider("Units Sold Change (%)",     -30, 50,  0)
            mktg  = st.slider("Marketing Spend Change (%)",-50, 100, 0)
        with s2:
            price = st.slider("Unit Price Change (%)",     -20, 30,  0)
            team  = st.slider("Sales Team Change (%)",     -20, 40,  0)
            seas  = st.slider("Seasonality Boost (%)",       0, 25,  0)

        factors = {
            "Discount":    1 + (-disc  * 0.015),
            "Units":       1 + (units  * 0.010),
            "Marketing":   1 + (mktg   * 0.004),
            "Price":       1 + (price  * 0.010),
            "Team":        1 + (team   * 0.006),
            "Seasonality": 1 + (seas   * 0.010),
        }
        adjusted = base
        for f in factors.values():
            adjusted *= f
        delta_pct = (adjusted - base) / base * 100

        r1, r2c, r3 = st.columns(3)
        r1.metric("Base Forecast",     "$" + f"{base:,.0f}")
        r2c.metric("Adjusted Forecast","$" + f"{adjusted:,.0f}",
                   str(round(delta_pct, 1)) + "%")
        r3.metric("Revenue Impact",    "$" + f"{adjusted-base:+,.0f}")

        impacts = [{"Factor": k, "Impact": round(base * (v - 1), 2)}
                   for k, v in factors.items()]
        wdf    = pd.DataFrame(impacts)
        colors_bar = ["#34d399" if v >= 0 else "#f87171" for v in wdf["Impact"]]
        fig_w  = go.Figure(go.Bar(
            x=wdf["Factor"], y=wdf["Impact"],
            marker_color=colors_bar,
            text=["$" + f"{v:+,.0f}" for v in wdf["Impact"]],
            textposition="outside",
        ))
        fig_w.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_C), height=300,
            yaxis=dict(gridcolor=GRID_C, tickprefix="$"),
            margin=dict(t=30))
        st.plotly_chart(fig_w, use_container_width=True)

    # ════ TAB 5: PDF ════
    with tab5:
        st.markdown('<div class="sec-title">📄 PDF Report Generator</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0d1f35;border:1px solid #1a3a5c;
                    border-radius:12px;padding:16px;margin-bottom:12px;">
          <b style="color:#60a5fa;">Report includes:</b><br><br>
          ✅ Executive KPI Summary<br>
          ✅ AI 90-Day Forecast with error bands<br>
          ✅ Auto-Generated Business Insights<br>
          ✅ Product and Region breakdown
        </div>""", unsafe_allow_html=True)

        if st.button("Generate and Download PDF", use_container_width=True):
            with st.spinner("Generating report..."):
                pdf  = generate_pdf(dff, monthly, forecast_df, r2, mae)
                fname = "sales_report_" + datetime.now().strftime("%Y%m%d_%H%M") + ".pdf"
                mime  = "application/pdf" if pdf[:4] == b"%PDF" else "text/plain"
                ext   = ".pdf" if mime == "application/pdf" else ".txt"
                st.download_button(
                    "Download Report",
                    data=pdf,
                    file_name=fname.replace(".pdf", ext),
                    mime=mime,
                    use_container_width=True)
                st.success("Ready! Click above to download.")

    # ════ TAB 6: MODEL ════
    with tab6:
        st.markdown('<div class="sec-title">🔬 Model Explainability</div>',
                    unsafe_allow_html=True)

        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Algorithm",      "Gradient Boosting")
        mm2.metric("R2 Accuracy",    str(round(r2, 4)))
        mm3.metric("MAE",            "$" + f"{mae:,.0f}")
        mm4.metric("Forecast Error", "+-$" + f"{mae:,.0f}")

        # Feature importance with human-readable names
        st.markdown('<div class="sec-title">What Drives the Predictions?</div>',
                    unsafe_allow_html=True)
        fi = pd.Series(model.feature_importances_, index=FEATS).sort_values(ascending=False)
        for feat, imp in fi.items():
            human = FEATURE_NAMES.get(feat, feat)
            pct   = round(imp * 100, 1)
            bar_w = int(pct * 4)
            st.markdown(
                '<div style="display:flex;align-items:center;padding:8px 0;'
                + 'border-bottom:1px solid #0f172a;">'
                + '<div style="font-size:0.85rem;color:#cbd5e1;width:220px;">' + human + "</div>"
                + '<div style="flex:1;margin:0 12px;background:#0f172a;border-radius:4px;height:8px;">'
                + '<div style="width:' + str(bar_w) + 'px;max-width:100%;height:100%;'
                + 'border-radius:4px;background:linear-gradient(90deg,#1d4ed8,#60a5fa);"></div>'
                + "</div>"
                + '<div style="font-size:0.8rem;color:#60a5fa;font-weight:600;">'
                + str(pct) + "%</div>"
                + "</div>",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("View Clean Dataset"):
            st.dataframe(df.head(50), use_container_width=True)
            st.download_button(
                "Download Clean CSV",
                data=df.to_csv(index=False).encode(),
                file_name="sales_clean.csv",
                mime="text/csv")

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#1e293b;font-size:0.7rem;padding:6px;letter-spacing:2px;">'
        + "SMART SALES FORECASTING v2.0  |  AI CHATBOT  |  AUTO INSIGHTS  |  WHAT-IF  |  PDF"
        + "</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()