import pandas as pd
from pathlib import Path

# --- Config ---
results_dir = Path('results')  # folder containing all results-*-*.csv files

# --- Collect all CSVs ---
metrics = {}  # { condition: { dataset: {metric: value} } }

for file in results_dir.glob("*.csv"):
    parts = file.stem.split("-")

    condition = '-'.join(parts[:-1])   # everything except last token
    dataset   = parts[-1].split('.')[0]

    df = pd.read_csv(file)
    if "correct" not in df.columns or "hit@1" not in df.columns:
        continue

    acc   = df["correct"].mean() * 100
    hit1  = df["hit@1"].mean() * 100
    hit3  = df["hit@3"].mean() * 100
    hit5  = df["hit@5"].mean() * 100

    metrics.setdefault(condition, {})
    metrics[condition][dataset] = {
        "Accuracy": acc,
        "Hit@1": hit1,
        "Hit@3": hit3,
        "Hit@5": hit5
    }

# --- Build tables ---
df = pd.concat(
    {cond: pd.DataFrame.from_dict(vals, orient='index')
     for cond, vals in metrics.items()},
    axis=0
)

df = df.sort_index()

print("\n=== Evaluation Results (% correct) ===")
print(df.round(3).to_string())
