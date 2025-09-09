# app.py
import io
import streamlit as st
import pandas as pd
import seaborn as sns

# plotting imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.helper_functions import (
    read_sheets,
    build_change_table_multi_list,
    pivots_by_metric,
)

# ---------- plotting helpers (added) ----------
def human_format(num):
    """Format numbers into K, M, B, etc."""
    num = float(num)
    if abs(num) >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:
        return f"{num/1_000_000:.0f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.0f}K"
    else:
        return str(int(num))

def plot_stacked_bars_from_change_table(
    change_table: pd.DataFrame,
    month_fmt: str = "%b-%Y",
    figsize: tuple = (10, 5),
    colors=None,
):
    """
    Draw one stacked bar chart per metric using the 'value' column from
    build_change_table_multi_list output.
    """
    # Normalize months for sorting
    change_table = change_table.copy()
    change_table["_month_dt"] = pd.to_datetime(change_table["month"]) + pd.offsets.MonthEnd(0)

    if colors is None:
        colors = plt.cm.tab10.colors

    figs = {}
    for metric, g in change_table.groupby("metric"):
        # rows=Month, cols=Partner, values=value
        mat = (
            g.pivot_table(index="_month_dt", columns="Partner", values="value", aggfunc="sum")
             .sort_index()
        )

        x = np.arange(len(mat.index))
        labels = [m.strftime(month_fmt) for m in mat.index]

        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(len(mat.index))

        for j, partner in enumerate(mat.columns):
            vals = mat[partner].fillna(0).values
            ax.bar(x, vals, bottom=bottom, label=partner, color=colors[j % len(colors)])
            bottom += vals

        ax.set_title(f"{metric} — Stacked by Partner")
        ax.set_xlabel("Month")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(title="Partner", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: human_format(val)))

        plt.tight_layout()
        figs[metric] = fig

    return figs
# ---------------------------------------------

st.set_page_config(page_title="Combine & Filter → Pivots", layout="wide")
st.title("Monthly Cross-Platform Metrics Dashboard")

uploaded = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

# Session state
"""
Session state (st.session_state) is like a persistent dictionary that remembers values between reruns for a given user session
"""
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "sheet_names" not in st.session_state:
    st.session_state.sheet_names = []
if "selected_sheets" not in st.session_state:
    st.session_state.selected_sheets = []
if "combined_df" not in st.session_state:
    st.session_state.combined_df = None

# Process file
if st.button("Process file", disabled=(uploaded is None)):
    buf = io.BytesIO(uploaded.getvalue())
    st.session_state.dataset = read_sheets(buf)
    st.session_state.sheet_names = list(st.session_state.dataset.keys())
    st.session_state.selected_sheets = st.session_state.sheet_names[:]
    st.success(f"Found {len(st.session_state.sheet_names)} sheet(s).")

# Select sheets
if st.session_state.sheet_names:
    chosen = st.multiselect(
        "Select the sheet(s) you're interested in:",
        options=st.session_state.sheet_names,
        default=st.session_state.selected_sheets,
    )
    st.session_state.selected_sheets = chosen

    # Combine selected → big DataFrame
    if st.session_state.selected_sheets:
        frames = []
        for name in st.session_state.selected_sheets:
            df = st.session_state.dataset[name].copy()
            df["Partner"] = name
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True, sort=True)
        combined["Month"] = pd.to_datetime(combined["Month"], errors="coerce") + pd.offsets.MonthEnd(0)
        st.session_state.combined_df = combined

        from src.helper_functions import get_month_labels

        month_label_map = get_month_labels(combined["Month"], fmt="%b %Y")  # e.g., "Jan 2020"
        month_choices = st.multiselect(
            "Select one or more months:",
            options=list(month_label_map.keys())[::-1],  # reverse chronological
            default=[list(month_label_map.keys())[-1]] if len(month_label_map) else [],
        )

        # Metrics selector (exclude Month & Partner)
        metric_options = [c for c in combined.columns if c not in ["Month", "Partner"]]
        metrics_chosen = st.multiselect(
            "Select metrics:",
            options=metric_options,
            default=metric_options,
        )

        # Build filtered long table, then wide pivots AND stacked bars
        if st.button("Build table, pivots & charts", disabled=(len(month_choices) == 0 or len(metrics_chosen) == 0)):
            change_tbl = build_change_table_multi_list(
                df=combined,
                target_months=month_choices,
                metrics=metrics_chosen,
                time_col="Month",
                brand_col="Partner",
                output_date_format="%m/%d/%Y",
            )

            st.subheader("Filtered change table (long format)")
            st.dataframe(change_tbl, use_container_width=True)

            st.subheader("Pivot tables (wide format) by metric")
            pivots = pivots_by_metric(change_tbl)
            for metric_name, pivot_df in pivots.items():
                with st.expander(f"📊 {metric_name}", expanded=False):
                    st.dataframe(
                        pivot_df.style.format({
                            "Value": "{:,.2f}",
                            "MoM %": "{:.2f}",
                            "YoY %": "{:.2f}",
                        }),
                        use_container_width=True
                    )

            st.subheader("Stacked bar charts (by metric)")
            n_partners = change_tbl["Partner"].nunique()
            cbf_colors = sns.color_palette("colorblind", n_colors=n_partners)
            figs = plot_stacked_bars_from_change_table(change_tbl, month_fmt="%b-%Y", colors=cbf_colors) # updated plots with color-blinded colors
            for metric_name, fig in figs.items():
                st.pyplot(fig, clear_figure=True)

    else:
        st.info("Select at least one sheet to combine.")
else:
    st.info("Upload a file and click **Process file** to see sheet names.")
