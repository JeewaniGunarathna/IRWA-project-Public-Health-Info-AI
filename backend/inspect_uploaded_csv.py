import pandas as pd
from pathlib import Path

# 👇 Use the actual relative path to your dataset
p = Path("app/agents/report_generator/data/timeseries_monthly.csv")
df = pd.read_csv(p)

print("✅ File loaded successfully!")
print("\nColumns:", list(df.columns))
print("\nSample rows:")
print(df.head(10))

print("\nUnique diseases:", df['disease'].unique()[:10])
if 'country_iso3' in df.columns:
    print("\nUnique country_iso3:", df['country_iso3'].unique()[:10])
if 'country' in df.columns:
    print("\nUnique country:", df['country'].unique()[:10])
if 'region' in df.columns:
    print("\nUnique region:", df['region'].unique()[:10])
