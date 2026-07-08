import io
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).parent / "data.csv"

st.set_page_config(
    page_title="Sales Analytics Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .metric-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 1rem;
            background: rgba(250, 250, 250, 0.04);
        }
        .small-caption {font-size: 0.85rem; opacity: 0.75;}
        .insight-box {
            padding: 1rem;
            border-radius: 0.8rem;
            border-left: 4px solid #6C63FF;
            background: rgba(108, 99, 255, 0.08);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_and_clean_data(uploaded_file=None):
    """Load the raw sales file and apply the cleaning steps used in the notebook."""
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_csv(DATA_PATH)

    original_rows = len(raw)
    null_rows = raw.isna().all(axis=1).sum()

    df = raw.dropna(how="all").copy()
    header_rows = df["Order Date"].astype(str).str.startswith("Or").sum()
    df = df[~df["Order Date"].astype(str).str.startswith("Or")].copy()

    df["Quantity Ordered"] = pd.to_numeric(df["Quantity Ordered"], errors="coerce")
    df["Price Each"] = pd.to_numeric(df["Price Each"], errors="coerce")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Quantity Ordered", "Price Each", "Order Date", "Purchase Address"])

    df["Sales"] = df["Quantity Ordered"] * df["Price Each"]
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.strftime("%b")
    df["Date"] = df["Order Date"].dt.date
    df["Hour"] = df["Order Date"].dt.hour
    df["Minute"] = df["Order Date"].dt.minute

    def extract_city(address: str) -> str:
        parts = str(address).split(",")
        if len(parts) < 3:
            return "Unknown"
        city = parts[1].strip()
        state = parts[2].strip().split(" ")[0]
        return f"{city}, {state}"

    df["City"] = df["Purchase Address"].apply(extract_city)

    bins = [0, 500, 1000, np.inf]
    labels = ["Low price", "Mid price", "Premium"]
    df["Price Category"] = pd.cut(df["Price Each"], bins=bins, labels=labels, include_lowest=True)

    cleaning_summary = {
        "Raw rows": original_rows,
        "Blank rows removed": int(null_rows),
        "Repeated header rows removed": int(header_rows),
        "Clean rows": len(df),
        "Unique orders": int(df["Order ID"].nunique()),
        "Date range": f"{df['Order Date'].min():%b %d, %Y} – {df['Order Date'].max():%b %d, %Y}",
    }
    return df, cleaning_summary


@st.cache_data(show_spinner=False)
def get_market_basket(df: pd.DataFrame, top_n: int = 15):
    duplicated = df[df["Order ID"].duplicated(keep=False)].copy()
    if duplicated.empty:
        return pd.DataFrame(columns=["Product Pair", "Times Sold Together"])

    duplicated["Grouped"] = duplicated.groupby("Order ID")["Product"].transform(lambda x: ",".join(x))
    baskets = duplicated[["Order ID", "Grouped"]].drop_duplicates()

    counter = Counter()
    for row in baskets["Grouped"]:
        products = row.split(",")
        counter.update(combinations(products, 2))

    return pd.DataFrame(
        [
            {"Product Pair": " + ".join(pair), "Times Sold Together": count}
            for pair, count in counter.most_common(top_n)
        ]
    )


def format_currency(value: float) -> str:
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:,.0f}"


def make_download(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ----------------------------
# Sidebar data + filters
# ----------------------------
st.sidebar.title("Sales Dashboard")
st.sidebar.caption("Built from a Python EDA notebook and converted into an interview-ready Streamlit app.")

uploaded_file = st.sidebar.file_uploader("Optional: upload another sales CSV", type=["csv"])
with st.spinner("Loading and cleaning sales data..."):
    df, cleaning_summary = load_and_clean_data(uploaded_file)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

city_options = sorted(df["City"].dropna().unique())
product_options = sorted(df["Product"].dropna().unique())
category_options = [str(x) for x in df["Price Category"].dropna().unique()]

selected_cities = st.sidebar.multiselect("City", city_options, default=city_options)
selected_products = st.sidebar.multiselect("Product", product_options, default=product_options)
selected_categories = st.sidebar.multiselect("Price category", category_options, default=category_options)

min_date = pd.to_datetime(df["Date"]).min().date()
max_date = pd.to_datetime(df["Date"]).max().date()
selected_dates = st.sidebar.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = df.copy()
if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered = filtered[(pd.to_datetime(filtered["Date"]).dt.date >= start_date) & (pd.to_datetime(filtered["Date"]).dt.date <= end_date)]
filtered = filtered[filtered["City"].isin(selected_cities)]
filtered = filtered[filtered["Product"].isin(selected_products)]
filtered = filtered[filtered["Price Category"].astype(str).isin(selected_categories)]

st.sidebar.download_button(
    "Download filtered data",
    data=make_download(filtered),
    file_name="filtered_sales_data.csv",
    mime="text/csv",
    use_container_width=True,
)

# ----------------------------
# Header
# ----------------------------
st.title("📊 Sales Analytics Portfolio Dashboard")
st.caption("Interactive sales performance dashboard for product, city, seasonality, and market-basket analysis.")

if filtered.empty:
    st.warning("No records match the current filters. Adjust the sidebar filters to continue.")
    st.stop()

# ----------------------------
# KPI Row
# ----------------------------
revenue = filtered["Sales"].sum()
orders = filtered["Order ID"].nunique()
units = filtered["Quantity Ordered"].sum()
avg_order_value = filtered.groupby("Order ID")["Sales"].sum().mean()
best_city = filtered.groupby("City")["Sales"].sum().idxmax()
best_product = filtered.groupby("Product")["Quantity Ordered"].sum().idxmax()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Revenue", format_currency(revenue))
kpi2.metric("Orders", f"{orders:,.0f}")
kpi3.metric("Units sold", f"{units:,.0f}")
kpi4.metric("Avg. order value", format_currency(avg_order_value))
kpi5.metric("Top city", best_city)

st.markdown(
    f"""
    <div class="insight-box">
        <b>Executive insight:</b> The selected data generated <b>{format_currency(revenue)}</b> across
        <b>{orders:,.0f}</b> orders. The strongest city is <b>{best_city}</b>, and the highest-volume product is
        <b>{best_product}</b>. Use the filters to explain how sales patterns change by city, product, price tier, and time period.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ----------------------------
# Tabs
# ----------------------------
overview_tab, product_tab, timing_tab, basket_tab, data_tab, resume_tab = st.tabs(
    ["Executive Overview", "Product & Pricing", "Timing Analysis", "Market Basket", "Data Quality", "Resume Story"]
)

with overview_tab:
    left, right = st.columns((1.2, 1))

    monthly = (
        filtered.groupby(["Month", "Month Name"], as_index=False)["Sales"]
        .sum()
        .sort_values("Month")
    )
    city_sales = filtered.groupby("City", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)

    with left:
        st.subheader("Revenue by month")
        fig = px.bar(
            monthly,
            x="Month Name",
            y="Sales",
            text_auto=".2s",
            labels={"Month Name": "Month", "Sales": "Sales ($)"},
            title="Monthly sales trend",
        )
        fig.update_layout(yaxis_tickprefix="$", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Revenue by city")
        fig = px.bar(
            city_sales,
            x="Sales",
            y="City",
            orientation="h",
            text_auto=".2s",
            labels={"Sales": "Sales ($)"},
            title="Top sales markets",
        )
        fig.update_layout(xaxis_tickprefix="$", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("City × month sales heatmap")
    heatmap = filtered.pivot_table(index="City", columns="Month", values="Sales", aggfunc="sum", fill_value=0)
    fig = px.imshow(
        heatmap,
        labels=dict(x="Month", y="City", color="Sales"),
        aspect="auto",
        title="Where and when revenue concentrates",
    )
    st.plotly_chart(fig, use_container_width=True)

with product_tab:
    col1, col2 = st.columns((1.1, 1))

    product_perf = (
        filtered.groupby("Product", as_index=False)
        .agg(Revenue=("Sales", "sum"), Units=("Quantity Ordered", "sum"), Avg_Price=("Price Each", "mean"), Orders=("Order ID", "nunique"))
        .sort_values("Revenue", ascending=False)
    )

    with col1:
        st.subheader("Product revenue leaderboard")
        fig = px.bar(
            product_perf.head(12),
            x="Revenue",
            y="Product",
            orientation="h",
            text_auto=".2s",
            hover_data=["Units", "Avg_Price", "Orders"],
            title="Top products by revenue",
        )
        fig.update_layout(xaxis_tickprefix="$", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Units sold vs. price")
        fig = px.scatter(
            product_perf,
            x="Avg_Price",
            y="Units",
            size="Revenue",
            hover_name="Product",
            labels={"Avg_Price": "Average price ($)", "Units": "Units sold"},
            title="Volume-price relationship",
        )
        fig.update_layout(xaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Revenue contribution by price category")
    cat_sales = filtered.groupby("Price Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    fig = px.treemap(cat_sales, path=["Price Category"], values="Sales", title="Sales by price tier")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed product table")
    st.dataframe(
        product_perf.assign(Revenue=product_perf["Revenue"].map(lambda x: f"${x:,.2f}"), Avg_Price=product_perf["Avg_Price"].map(lambda x: f"${x:,.2f}")),
        use_container_width=True,
        hide_index=True,
    )

with timing_tab:
    col1, col2 = st.columns(2)
    hourly = filtered.groupby("Hour", as_index=False).agg(Orders=("Order ID", "count"), Revenue=("Sales", "sum"))
    daily = filtered.groupby("Date", as_index=False)["Sales"].sum()

    with col1:
        st.subheader("Orders by hour")
        fig = px.line(hourly, x="Hour", y="Orders", markers=True, title="Peak order volume by hour")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue by hour")
        fig = px.line(hourly, x="Hour", y="Revenue", markers=True, title="Peak revenue by hour")
        fig.update_layout(yaxis_tickprefix="$")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily revenue trend")
    fig = px.line(daily, x="Date", y="Sales", title="Daily sales over time")
    fig.update_layout(yaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

with basket_tab:
    st.subheader("Products frequently sold together")
    basket = get_market_basket(filtered, top_n=15)

    if basket.empty:
        st.info("No multi-product orders exist for the current filters.")
    else:
        col1, col2 = st.columns((1.2, 1))
        with col1:
            fig = px.bar(
                basket,
                x="Times Sold Together",
                y="Product Pair",
                orientation="h",
                text="Times Sold Together",
                title="Top product combinations",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown(
                """
                **How to explain this in an interview**

                This is a simple market-basket analysis. I grouped products by shared order ID, generated two-product combinations, and counted which combinations appeared most often. A business team could use this for cross-sell bundles, checkout recommendations, or promotion design.
                """
            )
            st.dataframe(basket, hide_index=True, use_container_width=True)

with data_tab:
    st.subheader("Data cleaning summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Raw rows", f"{cleaning_summary['Raw rows']:,.0f}")
    c2.metric("Blank rows removed", f"{cleaning_summary['Blank rows removed']:,.0f}")
    c3.metric("Header rows removed", f"{cleaning_summary['Repeated header rows removed']:,.0f}")
    c4.metric("Clean rows", f"{cleaning_summary['Clean rows']:,.0f}")
    c5.metric("Unique orders", f"{cleaning_summary['Unique orders']:,.0f}")
    c6.metric("Date range", cleaning_summary["Date range"])

    st.markdown(
        """
        **Cleaning logic used:** removed fully blank rows, removed repeated CSV header rows, converted quantity and price to numeric values, parsed order timestamps, extracted city/state from address, calculated sales, and created time-based fields for analysis.
        """
    )

    st.subheader("Filtered data preview")
    st.dataframe(filtered.head(1000), use_container_width=True, hide_index=True)

with resume_tab:
    st.subheader("How to position this project")
    st.markdown(
        f"""
        **Project title:** Interactive Sales Analytics Dashboard  
        **Tools:** Python, Pandas, Streamlit, Plotly, market-basket analysis  
        **Dataset:** {cleaning_summary['Clean rows']:,.0f} cleaned transaction-level sales records  

        **Resume bullet options:**

        - Built an interactive Streamlit sales analytics dashboard using Python, Pandas, and Plotly to analyze {cleaning_summary['Clean rows']:,.0f}+ transaction records across products, cities, months, and order times.
        - Cleaned and transformed raw sales data by removing null rows and repeated headers, engineering revenue, city, month, hour, and price-tier features for business analysis.
        - Developed market-basket analysis to identify frequently bundled products, supporting cross-sell recommendations and promotion strategy.

        **Interview talking points:**

        1. I started with a raw transaction file and converted the notebook analysis into a reusable web app.
        2. I focused the dashboard around business questions: best month, best city, peak order hour, top products, and products sold together.
        3. I made it interactive so a recruiter or hiring manager can filter by city, product, price category, and date range.
        4. I included data-cleaning transparency so the project shows both analytics thinking and engineering discipline.
        """
    )

    st.info("Tip: Put the Streamlit app link in your resume under Projects, and link the GitHub repo next to it so interviewers can see both the product and the code.")
