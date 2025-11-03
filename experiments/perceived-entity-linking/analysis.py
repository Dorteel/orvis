import pandas as pd
from pathlib import Path

# --- Config ---
results_dir = Path()  # folder containing all results-*-*.csv files

# --- Collect all CSVs ---
data = {}
for file in results_dir.glob("results-*-*.csv"):
    # Extract condition and dataset from filename
    parts = file.stem.split("-")
    if len(parts) < 3:
        continue
    print(parts)
    condition = '-'.join(parts[1:-1])
    dataset = parts[-1].split('.')[0]
    #_, condition, dataset = parts[0], '-'.join(parts[1:-2]), parts[-1]  # handles dashes in dataset names

    # Load CSV and compute accuracy
    df = pd.read_csv(file)
    if "correct" not in df.columns:
        continue
    accuracy = df["correct"].mean() * 100  # percentage of True values

    # Store in nested dict
    data.setdefault(condition, {})[dataset] = accuracy

# --- Build DataFrame ---
table = pd.DataFrame(data).T.sort_index()  # conditions as rows
print("\n=== Accuracy Table (% correct per condition) ===")
print(table.round(4).to_string())