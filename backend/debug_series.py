import pandas as pd, numpy as np
from pathlib import Path

p = Path("app/agents/report_generator/data/timeseries_monthly.csv")
df = pd.read_csv(p)

# Normalize columns
df["disease_l"] = df["disease"].astype(str).str.strip().str.lower()

# Choose whichever region-like column exists
if "region" in df.columns:
    df["region_l"] = df["region"].astype(str).str.strip().str.lower()
elif "country" in df.columns:
    df["region_l"] = df["country"].astype(str).str.strip().str.lower()
elif "country_iso3" in df.columns:
    df["region_l"] = df["country_iso3"].astype(str).str.strip().str.lower()
else:
    raise ValueError("No region/country column found in the CSV")

# Filter for disease + region
sub = df[(df["disease_l"] == "dengue") & (df["region_l"] == "sri lanka")].copy()

print("rows:", len(sub))
if len(sub):
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    s = pd.to_numeric(sub["value"], errors="coerce")
    s = s[~s.isna()]
    print(
        "non-null values:", s.count(),
        "\nfirst date:", sub["date"].min(),
        "\nlast date:", sub["date"].max(),
    )
