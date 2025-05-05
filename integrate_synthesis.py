"""
Merge quantitative deltas with yearly policy label counts → tidy table
"""

import pandas as pd
from config import OUTPUT

def main():
    q_nat = pd.read_csv(OUTPUT / "ipbs_national_delta_08_10.csv").reset_index()
    q_nat.rename(columns={"index":"METRIC"}, inplace=True)

    pol = pd.read_csv(OUTPUT / "policy_classified.csv")
    pol["YEAR"] = pol["rule"].str.extract(r'(\d{4})').astype(int)
    yearly = pol.groupby(["YEAR","label"]).size().unstack(fill_value=0)
    yearly.columns = ["FFS_PARAS","VBC_PARAS"]
    yearly["VBC_SHARE"] = yearly["VBC_PARAS"] / yearly.sum(axis=1)

    yearly.to_csv(OUTPUT / "policy_yearly_counts.csv")
    q_nat.to_csv(OUTPUT / "quant_national_delta.csv", index=False)
    print("✅  Synthesis tables exported.")

if __name__ == "__main__":
    main()
