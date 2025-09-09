import pandas as pd
import seaborn as sns
def read_sheets(data_file): # function 1
    """
    Read all sheets from an Excel file into a dictionary.

    Parameters
    ----------
    data_file : Excel file path name

    Returns
    -------
    dataset : dict
        Keys are sheet names, values are DataFrames.
    """
    dataset = {}
    Excel_workbook = pd.ExcelFile(data_file)
    for partner in Excel_workbook.sheet_names:
        df = pd.read_excel(Excel_workbook, sheet_name = partner) # read each sheet into a DataFrame
        dataset[partner] = df # add the DataFrame to the dictionary
    return dataset

def user_interest(unfiltered_data, interested_data): # function 2
    """
    Filter the unfiltered_data dictionary to include only user-interested brands/sheets.

    Parameters
    ----------
    unfiltered_data : dict - output of function 1
        Keys are sheet names (brands), values are DataFrames.
    interested_data: list - user input of brands they are interested in

    Returns
    -------
    user_interested_data : dict
        Keys are user interested brands, values are DataFrames.
    """
    user_interested_data = {} 
    for brand, brand_data in unfiltered_data.items(): # loop through key and value in unfiltered_data dict
        if brand in interested_data: # if key matches user's interest
            user_interested_data[brand] = brand_data # add key and its dataframe to the dict
    return user_interested_data # output is a dict include all user interested key-value pairs

def combine_brand_dataframes(brand_dict): # function 3
    """
    Combine a dictionary of brand: DataFrame into one big DataFrame.
    Adds a 'brand' column to identify the source.

    Parameters
    ----------
    brand_dict : dict[str, pd.DataFrame] - output of function 2
        Keys are brand names, values are DataFrames with at least a 'time' column.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all metrics aligned and NA where missing.
    """
    frames = []
    for brand, df in brand_dict.items():
        df_copy = df.copy()
        df_copy["Partner"] = brand  # add brand column
        frames.append(df_copy)

    combined = pd.concat(frames, ignore_index=True, sort=True)
    return combined

# function 4a
from typing import List, Optional
def build_change_table_mdy_simple(
    df: pd.DataFrame,
    target_month: str,                # e.g., "8/31/2025" (any day in the month OK)
    metrics: Optional[List[str]] = None,
    time_col: str = "Month",
    brand_col: str = "Partner",
    output_date_format: str = "%m/%d/%Y",
) -> pd.DataFrame:
    """
    Build a change table for a single target month with MoM and YoY (%) changes.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least 'Month' and 'Partner' columns.
    target_month : str
        Target month in MM/DD/YYYY format (day can be any day in that month).
    metrics : list of str, optional
        List of metric column names the user wants. If None, display all numeric columns except brand_col.
    time_col : str
        Name of the time column. Default is 'Month'.
    brand_col : str
        Name of the brand column. Default is 'Partner'.
    output_date_format : str
        Format for the output month column. Default is '%m/%d/%Y'.

    Returns
    -------
    Returns a 6-column table of a single target month:
      month | brand | metric | value | month_over_month_pct | year_over_year_pct

    Assumes:
      - df already has NA for (brand, metric) combos that don't exist.
      - df contains a 'Month' (MM/DD/YYYY) and 'brand' columns.
    """

    out = df.copy()

    # 1) Normalize the time column to month-end for clean MoM/YoY math
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", infer_datetime_format=True)
    out[time_col] = out[time_col] + pd.offsets.MonthEnd(0)

    # 2) Parse target month and get previous and YoY comparison months
    tgt  = pd.to_datetime(target_month, errors="coerce", infer_datetime_format=True) + pd.offsets.MonthEnd(0)
    prev = tgt - pd.offsets.MonthEnd(1)
    yoy  = tgt - pd.offsets.MonthEnd(12)

    # 3) Choose metrics (if not provided, use all numeric columns except brand)
    if metrics is None:
        numeric_cols = out.select_dtypes("number").columns.tolist()
        metrics = [c for c in numeric_cols if c != brand_col]

    # 4) Reshape to long: (brand, month, metric, value)
    long = out[[brand_col, time_col] + metrics].melt(
        id_vars=[brand_col, time_col],
        var_name="metric",
        value_name="value",
    )

    # 5) Keep only rows for target, previous, and YoY months
    sub = long[long[time_col].isin({tgt, prev, yoy})]

    # 6) Pivot to have columns per month (index: brand, metric)
    pivot = sub.pivot_table(
        index=[brand_col, "metric"],
        columns=time_col,
        values="value",
        aggfunc="last",
    )

    # 7) Compute % changes safely (NA when missing/zero)
    def _pct(curr, base):
        return (curr / base - 1.0) if pd.notna(curr) and pd.notna(base) and base != 0 else pd.NA

    cur_vals  = pivot.get(tgt)
    prev_vals = pivot.get(prev)
    yoy_vals  = pivot.get(yoy)

    mom = cur_vals.combine(prev_vals, _pct) if prev_vals is not None else pd.Series(pd.NA, index=pivot.index)
    yoyp = cur_vals.combine(yoy_vals, _pct) if yoy_vals is not None else pd.Series(pd.NA, index=pivot.index)

    # 8) Keep only rows that actually exist at the target month
    if tgt in pivot.columns:
        mask_target = pivot[tgt].notna()
        pivot = pivot[mask_target]
    else:
        return pd.DataFrame(columns=["month", "Partner", "metric", "value", "month_over_month_pct", "year_over_year_pct"])

    # 9) Build final 6-column table (added 'value')
    res = (
        pd.DataFrame({
            "Partner": pivot.index.get_level_values(0),
            "metric": pivot.index.get_level_values(1),
            "value": pivot[tgt].loc[pivot.index],        # current month’s value
            "month_over_month_pct": mom.loc[pivot.index],
            "year_over_year_pct": yoyp.loc[pivot.index],
        })
        .reset_index(drop=True)
    )
    res.insert(0, "month", tgt.strftime(output_date_format))

    return res

# function 4b
def build_change_table_multi_list(
    df: pd.DataFrame,
    target_months: List[str],                 # e.g. ["03/31/2025","04/30/2025","05/31/2025"]
    metrics: Optional[List[str]] = None,
    time_col: str = "Month",
    brand_col: str = "Partner",
    output_date_format: str = "%m/%d/%Y",
) -> pd.DataFrame:
    """
    Build a change table for multiple target months with MoM and YoY (%) changes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least 'Month' and 'Partner' columns.
    target_months : list of str
        Target months in MM/DD/YYYY format (day can be any day in that month).
    metrics : list of str, optional
        List of metric column names the user wants. If None, display all numeric columns except brand_col.
    time_col : str
        Name of the time column. Default is 'Month'.
    brand_col : str
        Name of the brand column. Default is 'Partner'.
    output_date_format : str
        Format for the output month column. Default is '%m/%d/%Y'.

    Returns
    -------
    Returns a 6-column table of multiple target months:
      month | brand | metric | value | month_over_month_pct | year_over_year_pct
    """
    frames = []
    for tm in target_months:
        chunk = build_change_table_mdy_simple(
            df=df,
            target_month=tm,
            metrics=metrics,
            time_col=time_col,
            brand_col=brand_col,
            output_date_format=output_date_format,
        )
        # If a given month has no data, chunk may be empty; still append to keep behavior explicit
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["month", brand_col, "metric", "month_over_month_pct", "year_over_year_pct"])

    return pd.concat(frames, ignore_index=True)

# function 5
from typing import Dict
def pivots_by_metric(
    change_table: pd.DataFrame,
    month_col: str = "month",
    partner_col: str = "Partner",
    metric_col: str = "metric",
    value_col: str = "value",
    mom_col: str = "month_over_month_pct",
    yoy_col: str = "year_over_year_pct",
) -> Dict[str, pd.DataFrame]:
    """
    Build one pivot table per metric.

    Parameters
    ----------
    change_table : pd.DataFrame
        Output of build_change_table_multi_list in function 4b.
    month_col : str
        Name of the month column. Default is 'month'.
    partner_col : str
        Name of the brand column. Default is 'Partner'.
    metric_col : str
        Name of the metric column. Default is 'metric'.
    value_col : str
        Name of the value column. Default is 'value'.
    mom_col : str
        Name of the month-over-month % change column. Default is 'month_over_month_pct'.
    yoy_col : str
        Name of the year-over-year % change column. Default is 'year_over_year_pct'.
    
    Input (long table) must have columns:
      month | Partner | metric | value | month_over_month_pct | year_over_year_pct

    Returns
    -------
    A dictionary with key = metric, value = DataFrame:
        Structure of each DataFrame:
        index: MultiIndex (Partner, Month) with Month ascending
        columns: ["Value", "MoM %", "YoY %"]
        - Value is raw metric value
        - MoM % and YoY % are formatted as percentages (scaled by 100) with 2 decimals
    """
    df = change_table.copy()

    # Normalize month column to datetime for sorting
    month_dt = pd.to_datetime(df[month_col], errors="coerce", infer_datetime_format=True)
    df["_month_dt"] = month_dt
    # Format display as "Month-Year" (e.g., "June-2021")
    df["_month_disp"] = month_dt.dt.strftime("%B-%Y")

    out: Dict[str, pd.DataFrame] = {}
    for m, g in df.groupby(metric_col, dropna=False):
        sub = g[[partner_col, "_month_dt", "_month_disp", value_col, mom_col, yoy_col]].copy()
        sub = sub.sort_values("_month_dt")

        # Scale % changes
        sub[mom_col] = sub[mom_col] * 100
        sub[yoy_col] = sub[yoy_col] * 100

        # Set MultiIndex: Partner + Month
        sub = sub.set_index([partner_col, "_month_disp"], drop=True)
        sub.index.set_names([partner_col, "Month"], inplace=True)

        # Rename and keep desired columns
        sub = sub.rename(columns={
            value_col: "Value",
            mom_col: "MoM %",
            yoy_col: "YoY %"
        })[["Value", "MoM %", "YoY %"]]

        # Round % changes to 2 decimals (leave Value as is)
        sub[["MoM %", "YoY %", "Value"]] = sub[["MoM %", "YoY %", "Value"]].round(2)

        out[m] = sub
    return out

# function 6
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def human_format(num): # function for y-axis formatting (e.g., 200M instead of 200,000,000)
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
    month_fmt: str = "%b-%Y",   # x-axis label format
    figsize: tuple = (10, 5),
    colors=None,                # custom colors or defaults
):
    """
    Draw one stacked bar chart per metric using the 'value' column from
    build_change_table_multi_list output.

    Parameters
    ----------
    change_table : pd.DataFrame
        Output of build_change_table_multi_list in function 4b.
    month_fmt : str
        Format for x-axis month labels. Default is '%b-%Y' (e.g., 'Jan-2020').
    figsize : tuple
        Figure size. Default is (10, 5).
    colors : list or None
        List of colors to use for bars. If None, defaults to tab10 palette.

    Returns
    -------
    figs : dict
        Dictionary with key = metric, value = matplotlib Figure object.
    """
    # Normalize months for sorting
    change_table["_month_dt"] = pd.to_datetime(change_table["month"]) + pd.offsets.MonthEnd(0)

    # Default palette (tab10 has 10 good distinct colors)
    if colors is None:
        colors = plt.cm.Set2.colors 

    figs = {}
    for metric, g in change_table.groupby("metric"):
        # Pivot: rows=Month, cols=Partner, values=value
        mat = g.pivot_table(index="_month_dt", columns="Partner", values="value", aggfunc="sum").sort_index()

        # X positions and labels
        x = np.arange(len(mat.index))
        labels = [m.strftime(month_fmt) for m in mat.index]

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(len(mat.index))

        for j, partner in enumerate(mat.columns):
            vals = mat[partner].fillna(0).values
            color = colors[j % len(colors)]
            ax.bar(x, vals, bottom=bottom, label=partner, color=color)
            bottom += vals

        ax.set_title(f"{metric} — Stacked by Partner")
        ax.set_xlabel("Month")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(title="Partner", bbox_to_anchor=(1.02, 1), loc="upper left")

        # Use human_format y-axis labels
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: human_format(val)))

        plt.tight_layout()
        figs[metric] = fig

    return figs

# optional
def get_month_labels(series: pd.Series, fmt: str = "%b %Y"):
    """
    Convert datetime series to labels like 'Jan 2020' mapped to Timestamps.

    Parameters
    ----------
    series : pd.Series
        A datetime series (e.g., combined["Month"]).
    fmt : str
        Label format. Default '%b %Y' (Jan 2020).
        Use '%B %Y' for full month names (January 2020).

    Returns
    -------
    dict[str, Timestamp]
        Mapping of label -> datetime
    """
    months_sorted = series.dropna().sort_values().unique()
    return {pd.to_datetime(m).strftime(fmt): pd.to_datetime(m) for m in months_sorted}
