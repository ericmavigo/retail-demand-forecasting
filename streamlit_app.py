"""Interactive portfolio dashboard for the M5 demand forecasting project."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📦", layout="wide")
DATA = Path(__file__).parent / "app_data"

@st.cache_data
def load_data():
    names = ["daily_overview","weekly_overview","store_summary","category_summary","department_summary","product_summary","event_summary","model_metrics","store_model_comparison","forecast_daily","forecast_by_store","feature_importance","inventory_summary"]
    result = {name: pd.read_csv(DATA / f"{name}.csv") for name in names}
    for name in ["daily_overview","weekly_overview","forecast_daily"]:
        result[name]["date"] = pd.to_datetime(result[name]["date"])
    return result

d = load_data(); daily=d["daily_overview"]; weekly=d["weekly_overview"]
st.markdown("**DATA SCIENCE · FORECASTING · INVENTORY**")
st.title("Retail demand forecasting")
st.write("Five years of retail data transformed into business insight, a 28-day forecast and inventory decisions.")
overview, seasonality, portfolio, forecasting, inventory, methods = st.tabs(["Executive overview","Seasonality","Stores & products","Forecasting","Inventory","Methodology"])

with overview:
    best_store=d["store_summary"].iloc[0]; cols=st.columns(5)
    cols[0].metric("Units sold",f"{daily.units.sum()/1e6:.1f}M")
    cols[1].metric("Estimated revenue",f"${daily.estimated_revenue.sum()/1e6:.1f}M")
    cols[2].metric("Products",f"{len(d['product_summary']):,}")
    cols[3].metric("Stores",f"{len(d['store_summary']):,}")
    cols[4].metric("Top store",best_store.store_id)
    left,right=st.columns(2)
    with left: st.plotly_chart(px.bar(d["store_summary"].sort_values("estimated_revenue"),x="estimated_revenue",y="store_id",orientation="h",title="Estimated revenue by store"),width="stretch")
    with right: st.plotly_chart(px.treemap(d["department_summary"],path=["dept_id"],values="estimated_revenue",color="estimated_revenue",title="Revenue contribution by department"),width="stretch")
    yearly=daily.groupby("year",as_index=False).agg(units=("units","sum"),estimated_revenue=("estimated_revenue","sum"))
    st.plotly_chart(px.bar(yearly,x="year",y="estimated_revenue",text_auto=".3s",title="Estimated revenue by year"),width="stretch")
    st.caption("Revenue is estimated as units × weekly selling price. M5 has no basket ID, so a true average ticket cannot be calculated.")

with seasonality:
    fig=go.Figure([go.Scatter(x=daily.date,y=daily.units,name="Daily units",opacity=.3),go.Scatter(x=daily.date,y=daily.moving_average_28,name="28-day average",line=dict(width=3))]); fig.update_layout(title="Five-year demand history")
    st.plotly_chart(fig,width="stretch")
    st.plotly_chart(px.line(weekly[weekly.week_of_year<=52],x="week_of_year",y="units",color="year",title="Yearly demand curves by week"),width="stretch")
    left,right=st.columns(2); order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday=daily.groupby("weekday",as_index=False).units.mean(); weekday["weekday"]=pd.Categorical(weekday.weekday,order,ordered=True)
    with left: st.plotly_chart(px.bar(weekday.sort_values("weekday"),x="weekday",y="units",title="Average units by weekday"),width="stretch")
    with right: st.plotly_chart(px.line(daily.groupby("month",as_index=False).units.mean(),x="month",y="units",markers=True,title="Average daily demand by month"),width="stretch")
    events=d["event_summary"].dropna(subset=["unit_lift"]).sort_values("unit_lift",ascending=False).head(15)
    st.plotly_chart(px.bar(events.sort_values("unit_lift"),x="unit_lift",y="event_name",orientation="h",color="event_type",title="Event lift versus normal weekdays"),width="stretch")

with portfolio:
    left,right=st.columns(2)
    with left: st.plotly_chart(px.bar(d["category_summary"],x="cat_id",y="estimated_revenue",color="cat_id",title="Revenue by category"),width="stretch")
    with right: st.plotly_chart(px.scatter(d["store_summary"],x="units",y="estimated_revenue",text="store_id",size="estimated_revenue",title="Store productivity"),width="stretch")
    n=st.slider("Products to display",10,50,20); products=d["product_summary"].head(n)
    st.plotly_chart(px.bar(products.sort_values("estimated_revenue"),x="estimated_revenue",y="item_id",orientation="h",color="cat_id",title=f"Top {n} products"),width="stretch")
    pareto=d["product_summary"].copy(); pareto["product_share"]=(pareto.index+1)/len(pareto)
    st.plotly_chart(px.line(pareto,x="product_share",y="cumulative_revenue_share",title="Product revenue Pareto curve"),width="stretch")

with forecasting:
    metrics=d["model_metrics"].copy(); metrics["WAPE_percent"]=100*metrics.WAPE; best=metrics.sort_values("WAPE").iloc[0]; cols=st.columns(4)
    cols[0].metric("Best model",best.model.replace("_"," ")); cols[1].metric("WAPE",f"{best.WAPE_percent:.2f}%"); cols[2].metric("MAE",f"{best.MAE:.3f}"); cols[3].metric("RMSSE",f"{best.RMSSE:.3f}")
    st.plotly_chart(px.bar(metrics.sort_values("WAPE_percent",ascending=False),x="WAPE_percent",y="model",orientation="h",text_auto=".2f",title="28-day holdout accuracy"),width="stretch")
    fc=d["forecast_daily"].melt("date",var_name="series",value_name="units")
    st.plotly_chart(px.line(fc,x="date",y="units",color="series",title="Actual versus forecast demand"),width="stretch")
    left,right=st.columns(2)
    with left: st.plotly_chart(px.bar(d["forecast_by_store"].sort_values("WAPE"),x="store_id",y="WAPE",title="Error by store"),width="stretch")
    with right: st.plotly_chart(px.bar(d["feature_importance"].head(12).sort_values("importance"),x="importance",y="feature",orientation="h",title="LightGBM feature importance"),width="stretch")
    st.info("The 25% LightGBM hybrid improves the baseline slightly. The weaker pure LightGBM result remains visible for transparency.")

    st.divider()
    st.subheader("Store-total model comparison")
    st.write("A second comparison forecasts each store's combined demand across all products and categories. It uses the same 28-day holdout for every model.")
    store_metrics=d["store_model_comparison"].copy()
    store_metrics["WAPE_percent"]=100*store_metrics.WAPE
    best_store_model=store_metrics.sort_values("WAPE").iloc[0]
    c1,c2,c3=st.columns(3)
    c1.metric("Best store-total model",best_store_model.model)
    c2.metric("Store-total WAPE",f"{best_store_model.WAPE_percent:.2f}%")
    c3.metric("Store-total MAE",f"{best_store_model.MAE:,.2f} units/day")
    st.plotly_chart(px.bar(store_metrics.sort_values("WAPE_percent",ascending=False),x="WAPE_percent",y="model",orientation="h",text_auto=".2f",title="Store-total model accuracy · same 28-day holdout",labels={"WAPE_percent":"WAPE (%)","model":"Model"}),width="stretch")
    st.dataframe(store_metrics[["model","MAE","WAPE","Bias"]].style.format({"MAE":"{:,.2f}","WAPE":"{:.2%}","Bias":"{:+.2%}"}),hide_index=True,width="stretch")
    st.caption("This comparison aggregates all products and categories within each of the 10 stores. Its WAPE and MAE are not directly comparable with the item-store results above. Prophet uses weekly/yearly seasonality and M5 event dates; NeuralProphet uses weekly/yearly seasonality and the same events; SARIMAX uses weekly seasonality, annual Fourier terms, weekdays and M5 event indicators.")

with inventory:
    inv=d["inventory_summary"]
    st.plotly_chart(px.bar(inv.melt("method",var_name="outcome",value_name="units"),x="method",y="units",color="outcome",barmode="group",title="Inventory trade-off"),width="stretch")
    reduction=1-inv.iloc[1].stockout_units/inv.iloc[0].stockout_units; st.metric("Estimated stockout-unit reduction",f"{reduction:.1%}")
    st.write("The simulator uses the 28-day order quantity, a 95% service assumption and a seven-day lead time to estimate safety stock and reorder points.")

with methods:
    st.subheader("Workflow")
    st.markdown("1. Validate sales, calendar and price keys.\n2. Separate valid zero demand from quality problems.\n3. Estimate revenue as units × weekly price.\n4. Preserve a 28-day future holdout.\n5. Compare item-store models and store-total statistical/ML forecasts on a shared holdout.\n6. Translate forecasts into inventory outcomes.")
    st.subheader("Limitations")
    st.write("M5 contains aggregated demand, not transactions, customers, inventory-on-hand or purchase orders. Revenue and inventory outcomes are analytical estimates.")
    st.caption("Built by Eric Villegas · Data Science & AI portfolio project")
