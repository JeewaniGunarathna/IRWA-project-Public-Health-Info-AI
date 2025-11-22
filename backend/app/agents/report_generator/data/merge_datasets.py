# backend/app/agents/report_generator/data/merge_datasets.py
import pandas as pd
from pathlib import Path

# Resolve paths relative to THIS file (not the working directory)
THIS_FILE = Path(__file__).resolve()
AGENT_DIR = THIS_FILE.parents[1]                  # .../report_generator
PARTS_DIR = AGENT_DIR / "data_parts"              # folder where you placed the small CSVs
OUT_DIR   = AGENT_DIR / "data"                    # merged output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Gather all CSV files
parts = sorted(PARTS_DIR.glob("*.csv"))
print(f"Merging {len(parts)} CSV files from: {PARTS_DIR}")

if not parts:
    raise SystemExit("No CSV files found. Check that your files are in: "
                     f"{PARTS_DIR} and have the .csv extension.")

# Read + concatenate
df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)

# Drop duplicates and empty rows
for col in ("disease", "date", "value"):
    if col not in df.columns:
        raise SystemExit(f"Expected column '{col}' not found in inputs.")
df.dropna(subset=["disease", "date", "value"], inplace=True)
df.drop_duplicates(inplace=True)

# Sort for readability (only if columns exist)
sort_cols = [c for c in ["disease", "country_iso3", "date"] if c in df.columns]
if sort_cols:
    df.sort_values(by=sort_cols, inplace=True)

# Save final merged CSV
out_path = OUT_DIR / "timeseries_monthly.csv"
df.to_csv(out_path, index=False)

print(f"✅ Merged dataset saved to: {out_path}")
print(f"Total rows: {len(df)}")
