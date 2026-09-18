"""Single-page executive dashboard for the M5 demand forecasting portfolio."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Retail Forecasting · M5", page_icon="📦", layout="wide")
DATA = Path(__file__).parent / "app_data"

@st.cache_data
def load_data():
    names = ["daily_overview", "weekly_overview", "store_summary", "category_summary",
             "department_summary", "product_summary", "event_summary", "model_metrics",
             "store_model_comparison", "forecast_daily", "forecast_by_store",
             "feature_importance", "inventory_summary"]
    result = {name: pd.read_csv(DATA / f"{name}.csv") for name in names}
    for name in ["daily_overview", "weekly_overview", "forecast_daily"]:
        result[name]["date"] = pd.to_datetime(result[name]["date"])
    return result

d = load_data()
daily, weekly = d["daily_overview"], d["weekly_overview"]
item_metrics = d["model_metrics"].copy()
store_metrics = d["store_model_comparison"].copy()
item_best = item_metrics.sort_values("WAPE").iloc[0]
store_best = store_metrics.sort_values("WAPE").iloc[0]
baseline = item_metrics.loc[item_metrics.model.eq("mean_last_28")].iloc[0]
inv = d["inventory_summary"]
stockout_reduction = 1 - inv.iloc[1].stockout_units / inv.iloc[0].stockout_units
excess_change = inv.iloc[1].excess_units / inv.iloc[0].excess_units - 1

st.markdown("""
<style>
.block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem;}
.hero {background:linear-gradient(120deg,#10283F 0%,#165C58 100%); padding:2.3rem 2.6rem; border-radius:20px; color:white; margin:0.5rem 0 1.3rem;}
.hero-kicker {font-size:.78rem; letter-spacing:.16em; font-weight:700; color:#9FE4D2; text-transform:uppercase; margin-bottom:.65rem;}
.hero h1 {color:white; font-size:2.45rem; line-height:1.12; margin:.15rem 0 .75rem;}
.hero p {color:#E5F2F1; font-size:1.06rem; max-width:900px; margin-bottom:.4rem;}
.section-kicker {color:#167D72; font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; font-weight:750; margin:1.4rem 0 .2rem;}
div[data-testid="stMetric"] {background:rgba(128,145,160,.12); border:1px solid rgba(128,145,160,.24); padding:1rem 1.1rem; border-radius:14px;}
div[data-testid="stMetricLabel"] p {font-size:.9rem;}
div[data-testid="stMetricValue"] {font-size:1.65rem;}
</style>
<div class="hero">
  <div class="hero-kicker">Retail analytics · demand forecasting · inventory decisions</div>
  <h1>From five years of sales to a clearer buying decision.</h1>
  <p>Explore the demand patterns, compare forecasting approaches on the same 28-day test, and see how forecast quality changes the inventory trade-off.</p>
</div>
""", unsafe_allow_html=True)
st.caption("M5 Forecasting · Accuracy | 10 stores · 3,049 products · 30,490 item-store series · daily data from 2011 to 2016")

st.markdown('<div class="section-kicker">Executive readout</div>', unsafe_allow_html=True)
hero_cols = st.columns(4)
hero_cols[0].metric("Units sold", f"{daily.units.sum()/1e6:.1f}M")
hero_cols[1].metric("Estimated revenue", f"${daily.estimated_revenue.sum()/1e6:.1f}M")
hero_cols[2].metric("Item-store winner", item_best.model.replace("_", " "), f"{item_best.WAPE:.2%} WAPE")
hero_cols[3].metric("Store-total winner", store_best.model, f"{store_best.WAPE:.2%} WAPE")
st.info(f"**Two answers at two levels:** the item-store hybrid leads on individual series ({item_best.WAPE:.2%} WAPE), while Global LightGBM leads after all products are added up inside each store ({store_best.WAPE:.2%}). These scores have different aggregation levels, so compare models within each panel.")

st.markdown('<div class="section-kicker">01 · The forecasting challenge</div>', unsafe_allow_html=True)
st.header("Which model sees demand most clearly?")
st.write("Every model uses the same 28-day future holdout. The left panel scores the detailed item-store forecasts; the right panel scores total demand for each of the 10 stores.")
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Item × store")
    plot_item = item_metrics.assign(WAPE_percent=100*item_metrics.WAPE).sort_values("WAPE_percent", ascending=True)
    fig = px.bar(plot_item, x="WAPE_percent", y="model", orientation="h", text="WAPE_percent",
                 title="Forecast error across item-store series", labels={"WAPE_percent":"WAPE (%)", "model":""},
                 color="WAPE_percent", color_continuous_scale=["#8DD3C7", "#167D72"])
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, height=370, margin=dict(l=8,r=45,t=55,b=15))
    st.plotly_chart(fig, width="stretch")
    item_table = item_metrics[["model", "MAE", "WAPE", "RMSSE", "Bias"]].copy()
    item_table["model"] = item_table.model.str.replace("_", " ", regex=False)
    item_table["MAE"] = item_table.MAE.map(lambda x: f"{x:.3f}")
    for col in ["WAPE", "Bias"]: item_table[col] = item_table[col].map(lambda x: f"{x:+.2%}" if col == "Bias" else f"{x:.2%}")
    item_table["RMSSE"] = item_table.RMSSE.map(lambda x: f"{x:.3f}")
    st.dataframe(item_table, hide_index=True, width="stretch")
with right:
    st.subheader("Total demand by store")
    plot_store = store_metrics.assign(WAPE_percent=100*store_metrics.WAPE).sort_values("WAPE_percent", ascending=True)
    fig = px.bar(plot_store, x="WAPE_percent", y="model", orientation="h", text="WAPE_percent",
                 title="Forecast error after aggregating each store", labels={"WAPE_percent":"WAPE (%)", "model":""},
                 color="WAPE_percent", color_continuous_scale=["#8DD3C7", "#167D72"])
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, height=370, margin=dict(l=8,r=45,t=55,b=15))
    st.plotly_chart(fig, width="stretch")
    store_table = store_metrics[["model", "MAE", "WAPE", "Bias"]].copy()
    store_table["MAE"] = store_table.MAE.map(lambda x: f"{x:,.2f}")
    store_table["WAPE"] = store_table.WAPE.map(lambda x: f"{x:.2%}")
    store_table["Bias"] = store_table.Bias.map(lambda x: f"{x:+.2%}")
    st.dataframe(store_table, hide_index=True, width="stretch")
st.caption("WAPE is the total absolute forecast error divided by total actual demand; lower is better. MAE is average units of error per day. RMSSE appears only where calculated. The different aggregation levels make the two panels non-comparable to each other.")

st.markdown('<div class="section-kicker">02 · What happened in the test window</div>', unsafe_allow_html=True)
st.header("A 28-day forecast, side by side with actual sales")
forecast = d["forecast_daily"].rename(columns={"actual_units":"Actual demand", "baseline_units":"28-day baseline", "model_units":"Hybrid LightGBM"})
fig = go.Figure()
for name, color, width in [("Actual demand", "#17324D", 3), ("28-day baseline", "#9AA6B2", 2), ("Hybrid LightGBM", "#E08A36", 2.5)]:
    fig.add_trace(go.Scatter(x=forecast.date, y=forecast[name], mode="lines+markers", name=name, line=dict(color=color,width=width), marker=dict(size=5)))
fig.update_layout(title="Daily units · actual vs forecast", xaxis_title="Date", yaxis_title="Units", hovermode="x unified", height=430, legend=dict(orientation="h",y=1.12))
st.plotly_chart(fig, width="stretch")
actual_total = forecast["Actual demand"].sum()
pred_total = forecast["Hybrid LightGBM"].sum()
fc1,fc2,fc3 = st.columns(3)
fc1.metric("Actual units in test", f"{actual_total:,.0f}")
fc2.metric("Hybrid forecast", f"{pred_total:,.0f}", f"{pred_total/actual_total-1:+.2%} vs actual")
fc3.metric("Mean daily forecast error", f"{item_best.MAE:.3f} units / series / day", "item-store level")

st.markdown('<div class="section-kicker">03 · Inventory consequences</div>', unsafe_allow_html=True)
st.header("A forecast matters when it changes the shelf decision")
inventory_left, inventory_right = st.columns([1.25,1], gap="large")
with inventory_left:
    inv_plot=inv.rename(columns={"method":"Planning approach", "stockout_units":"Estimated shortage units", "excess_units":"Estimated excess units"})
    fig=px.bar(inv_plot.melt("Planning approach",var_name="Outcome",value_name="Units"),x="Planning approach",y="Units",color="Outcome",barmode="group",text_auto=",.0f",title="Simulated trade-off over the 28-day holdout",color_discrete_sequence=["#C65D4B","#E7B55A"])
    fig.update_layout(height=380,legend_title_text="",margin=dict(t=55,b=15))
    st.plotly_chart(fig,width="stretch")
with inventory_right:
    st.metric("Fewer shortage units",f"{stockout_reduction:.1%}","vs baseline plan")
    st.metric("Change in excess units",f"{excess_change:+.1%}","trade-off vs baseline")
    st.write("In this simple simulation, the hybrid reduces estimated shortage units while holding slightly more excess stock. These are scenario estimates, not observed warehouse outcomes.")

st.markdown('<div class="section-kicker">04 · The demand story</div>', unsafe_allow_html=True)
st.header("Five years of seasonality, calendar effects and store differences")
fig=go.Figure([go.Scatter(x=daily.date,y=daily.units,name="Daily units",opacity=.25,line=dict(color="#73909E")),go.Scatter(x=daily.date,y=daily.moving_average_28,name="28-day moving average",line=dict(color="#167D72",width=3))])
fig.update_layout(title="Demand history · 2011–2016",xaxis_title="Date",yaxis_title="Units per day",height=390,hovermode="x unified")
st.plotly_chart(fig,width="stretch")
season_a, season_b = st.columns(2, gap="large")
with season_a:
    fig=px.line(weekly[weekly.week_of_year<=52],x="week_of_year",y="units",color="year",title="How the seasonal pattern shifts by year",labels={"week_of_year":"Week of year","units":"Units","year":"Year"})
    fig.update_layout(height=370)
    st.plotly_chart(fig,width="stretch")
with season_b:
    order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday=daily.groupby("weekday",as_index=False).units.mean(); weekday["weekday"]=pd.Categorical(weekday.weekday,order,ordered=True)
    fig=px.bar(weekday.sort_values("weekday"),x="weekday",y="units",title="Average demand by day of week",labels={"weekday":"Day","units":"Average units / day"},color_discrete_sequence=["#167D72"])
    fig.update_layout(height=370)
    st.plotly_chart(fig,width="stretch")
month, events = st.columns(2, gap="large")
with month:
    monthly=daily.groupby("month",as_index=False).units.mean()
    st.plotly_chart(px.line(monthly,x="month",y="units",markers=True,title="Average units by month",labels={"month":"Month","units":"Average units / day"}),width="stretch")
with events:
    event_data=d["event_summary"].dropna(subset=["unit_lift"]).sort_values("unit_lift",ascending=False).head(12).sort_values("unit_lift")
    st.plotly_chart(px.bar(event_data,x="unit_lift",y="event_name",orientation="h",color="event_type",title="Calendar events with the largest demand lift",labels={"unit_lift":"Lift vs normal weekday","event_name":"Event","event_type":"Type"}),width="stretch")

st.markdown('<div class="section-kicker">05 · Where demand comes from</div>', unsafe_allow_html=True)
st.header("Find the stores, categories and products that drive volume")
store_col, cat_col = st.columns(2,gap="large")
with store_col:
    st.plotly_chart(px.bar(d["store_summary"].sort_values("units"),x="units",y="store_id",orientation="h",text="units",title="Units sold by store",labels={"units":"Units sold","store_id":"Store"},color_discrete_sequence=["#167D72"]),width="stretch")
with cat_col:
    st.plotly_chart(px.bar(d["category_summary"].sort_values("units"),x="units",y="cat_id",orientation="h",text="units",title="Units sold by category",labels={"units":"Units sold","cat_id":"Category"},color="cat_id"),width="stretch")
cat_options=["All categories"]+sorted(d["product_summary"].cat_id.dropna().unique().tolist())
selected_category=st.selectbox("Explore leading products",cat_options)
products=d["product_summary"] if selected_category=="All categories" else d["product_summary"].loc[d["product_summary"].cat_id.eq(selected_category)]
number=st.slider("Products to show",5,30,15)
products=products.sort_values("units",ascending=False).head(number).sort_values("units")
st.plotly_chart(px.bar(products,x="units",y="item_id",orientation="h",color="cat_id",text="units",title=f"Top {len(products)} products by units sold",labels={"units":"Units sold","item_id":"Product","cat_id":"Category"}),width="stretch")

with st.expander("Method, model settings and limitations"):
    st.markdown("""
    **Validation.** All candidates are scored on the same final 28 days, `d_1914`–`d_1941`. The product-level work uses item-store demand; the separate classical time-series comparison sums demand to the 10 store totals to keep runtimes practical.

    **Calendar features.** Prophet and NeuralProphet use weekly and yearly seasonality plus event dates from the M5 calendar. SARIMAX uses weekly seasonal autoregression, annual Fourier terms, weekday indicators and M5 event indicators. The item-store models use a global LightGBM forecast blended with a moving-average baseline.

    **Interpretation.** WAPE and MAE across aggregation levels are not comparable. Lower error is better within the same panel. Revenue is estimated from units × selling price; M5 has no transaction or basket identifier, inventory-on-hand, or purchase-order history. Inventory results are scenario estimates rather than observed operational outcomes.
    """)
    st.markdown("[Open the executable forecasting notebook](https://colab.research.google.com/github/ericmavigo/retail-demand-forecasting/blob/main/notebooks/03_lightgbm_forecasting.ipynb) · [Explore the repository on GitHub](https://github.com/ericmavigo/retail-demand-forecasting)")

st.caption("Built by Eric Villegas · Data science and operations portfolio · M5 Forecasting dataset")
