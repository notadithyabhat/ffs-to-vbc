"""
Generate publication‑ready plots for report figs directory
"""

import pandas as pd, matplotlib.pyplot as plt
from config import OUTPUT

plt.rcParams.update({"figure.autolayout": True, "font.size": 10})

def bar_national():
    df = pd.read_csv(OUTPUT / "quant_national_delta.csv")
    df["LABEL"] = df["METRIC"].str.replace("_BENE_CNT","")
    plt.figure(figsize=(5,3))
    plt.bar(df["LABEL"], df["PERCENT"])
    plt.ylabel("% change 08→10")
    plt.title("National utilisation change")
    plt.xticks(rotation=45)
    plt.savefig(OUTPUT / "figs/national_deltas.png", dpi=300)

def policy_share():
    yr = pd.read_csv(OUTPUT / "policy_yearly_counts.csv", index_col="YEAR")
    plt.figure(figsize=(4,3))
    plt.plot(yr.index, yr["VBC_SHARE"]*100, marker="o")
    plt.ylabel("Share of policy text labelled VBC (%)")
    plt.title("VBC language trajectory 2008‑2010")
    plt.ylim(0,100)
    plt.savefig(OUTPUT / "figs/policy_share.png", dpi=300)

def main():
    bar_national()
    policy_share()
    print("✅  Figures written to outputs/figs")

if __name__ == "__main__":
    main()
