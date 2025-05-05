"""
Expanded quantitative ETL:
* processes full PSBS file (stream/Dask)
* builds 2008 vs 2010 delta tables
* aggregates by HRR & facility type
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
from config import *

# ---------- helper cleaning ----------
def _clean_cols(df):
    df.columns = [c.upper().strip() for c in df.columns]
    return df

def load_ipbs(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"PROVIDER_ID": str})
    df = _clean_cols(df)
    return df

def load_psbs(path: pathlib.Path) -> dd.DataFrame:
    dtype_fix = {"PROVIDER_ID": "object",
                 "NPPES_PROVIDER_MAILING_ZIP": "object",
                 "NPPES_PROVIDER_PRACTICE_ZIP": "object",
                 "PROVIDER_SPECIALTY": "object"}
    ddf = dd.read_csv(path,
                      blocksize=DASK_BLOCKSIZE,
                      dtype=dtype_fix,
                      assume_missing=True)
    ddf = ddf.map_partitions(_clean_cols)
    # provider id normalisation
    ddf["PROVIDER_ID"] = (
        ddf["PROVIDER_ID"]
        .astype(float)  # remove .0 suffix
        .astype("Int64")
        .astype(str)
    )
    return ddf

def main() -> None:
    print("🔄  Loading IPBS 2008/2010 …")
    ip08, ip10 = load_ipbs(IPBS_08), load_ipbs(IPBS_10)

    print("🔄  Loading PSBS 2010 (full or large sample) …")
    psbs_dd = load_psbs(PSBS_10)

    # ---------- 1. provider‑level PSBS aggregation ----------
    agg_cols = {"BENE_CNT": "sum", "TOTAL_SRVC_CNT": "sum"}
    psbs_agg = (
        psbs_dd.groupby("PROVIDER_ID")
        .agg(agg_cols)
        .compute()
        .reset_index()
    )
    psbs_agg.to_csv(OUTPUT / "psbs_provider_agg_2010.csv", index=False)

    # ---------- 2. HRR roll‑up ----------
    zip_hrr = pd.read_csv(ZIP_HRR_XW, dtype=str)
    psbs_zip = psbs_dd.merge(zip_hrr[["ZIP", "HRR"]], left_on="NPPES_PROVIDER_PRACTICE_ZIP",
                             right_on="ZIP", how="left")
    hrr_agg = (
        psbs_zip.groupby("HRR")[["BENE_CNT", "TOTAL_SRVC_CNT"]]
        .sum()
        .compute()
        .reset_index()
    )
    hrr_agg.to_csv(OUTPUT / "psbs_hrr_agg_2010.csv", index=False)

    # ---------- 3. 2008 vs 2010 delta (inpatient) ----------
    metrics = ["AGE_LESS_65_BENE_CNT","AGE_65_69_BENE_CNT","READMITS"]
    ip08_tot = ip08[metrics].sum()
    ip10_tot = ip10[metrics].sum()
    delta = (ip10_tot - ip08_tot).to_frame("ABS_DIFF")
    delta["PERCENT"] = (delta["ABS_DIFF"] / ip08_tot)*100
    delta.to_csv(OUTPUT / "ipbs_national_delta_08_10.csv")

    print("✅  Quantitative aggregation complete → outputs/")

if __name__ == "__main__":
    main()
