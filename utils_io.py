"""
utils_io.py — shared read/write helpers for the summary-statistics table format.

Every script that touches per-site or pooled summary stats should go through
these functions, so the CSV schema (column names, order, dtypes) stays
consistent everywhere: simulator output, real-data preprocessing, federated
aggregation, and diagnostics all read/write the same shape of table.
"""

import os
import csv
import pandas as pd

from config import (
    SUMMARY_STATS_COLUMNS,
    SITES_DIR,
    SITE_FILE_TEMPLATE,
)


# ----------------------------------------------------------------------------
# Core read/write
# ----------------------------------------------------------------------------
def write_summary_stats(df: pd.DataFrame, path: str) -> None:
    """
    Write a summary-stats DataFrame to CSV, enforcing column order and presence.
    Raises ValueError if required columns are missing.
    """
    missing = [c for c in SUMMARY_STATS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns before write: {missing}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[SUMMARY_STATS_COLUMNS].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def read_summary_stats(path: str) -> pd.DataFrame:
    """
    Read a summary-stats CSV, enforcing the expected schema.
    Raises ValueError if required columns are missing.
    """
    df = pd.read_csv(path)
    missing = [c for c in SUMMARY_STATS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df[SUMMARY_STATS_COLUMNS]


# ----------------------------------------------------------------------------
# Per-site convenience wrappers
# ----------------------------------------------------------------------------
def site_path(site_id) -> str:
    """Return the standard file path for a given site_id."""
    return os.path.join(SITES_DIR, SITE_FILE_TEMPLATE.format(site_id=site_id))


def write_site_data(site_id, df: pd.DataFrame) -> None:
    """Write one site's summary stats to its standard location."""
    write_summary_stats(df, site_path(site_id))


def read_site_data(site_id) -> pd.DataFrame:
    """Read one site's summary stats from its standard location."""
    return read_summary_stats(site_path(site_id))


def read_all_sites(n_sites: int) -> dict:
    """
    Read summary stats for sites 0..n_sites-1.
    Returns dict: {site_id: DataFrame}.
    """
    return {site_id: read_site_data(site_id) for site_id in range(n_sites)}


def concat_all_sites(n_sites: int) -> pd.DataFrame:
    """
    Read and concatenate all sites into one DataFrame (i.e. the 'centralized'
    view of federated data). Useful for the centralized baseline and for
    sanity checks — never used inside the actual federated aggregation logic,
    which must only see one site at a time.
    """
    frames = [read_site_data(site_id) for site_id in range(n_sites)]
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# Results (non-summary-stats) helpers — for CI tables, sweep results, etc.
# ----------------------------------------------------------------------------
def write_results_table(df: pd.DataFrame, path: str) -> None:
    """Write an arbitrary results DataFrame (no fixed schema) to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def read_results_table(path: str) -> pd.DataFrame:
    """Read an arbitrary results CSV back in."""
    return pd.read_csv(path)
