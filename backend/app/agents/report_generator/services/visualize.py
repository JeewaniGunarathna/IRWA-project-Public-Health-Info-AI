from pathlib import Path
from typing import List, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime

def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in text).strip("-")

def make_line_chart(
    timeseries: List[Dict],
    disease: str,
    region: str,
    date_from: str,
    date_to: str,
    charts_dir: str,
) -> str:
    """
    Creates a line chart PNG from a list of {date, value} dicts.
    Returns the relative file path to the saved image.
    """
    Path(charts_dir).mkdir(parents=True, exist_ok=True)

    # Sort by date just in case
    ts_sorted = sorted(timeseries, key=lambda x: x["date"])

    # x, y data
    x = [datetime.datetime.fromisoformat(p["date"]) for p in ts_sorted]
    y = [float(p["value"]) for p in ts_sorted]

    # filename
    fname = f"{slugify(disease)}_{slugify(region)}_{date_from.replace('-','')}_{date_to.replace('-','')}.png"
    out_path = Path(charts_dir) / fname

    # draw
    plt.figure(figsize=(8, 4.2))
    plt.plot(x, y, marker="o")
    plt.title(f"{disease} in {region} • {date_from} → {date_to}")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

    # return a **relative** path so UI can load it
    return f"/agents/report_generator/artifacts/charts/{fname}"
