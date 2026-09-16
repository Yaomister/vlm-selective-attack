import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def flipped(g):
    clean = g["safety_clean"].str.lower().str.startswith("yes")
    pert = g["safety_perturbed"].str.lower().str.startswith("yes")
    return clean != pert

METRICS = {
    "attack_success": lambda g: flipped(g).mean(),
    "flip_count":     lambda g: flipped(g).sum(),
    "safety_shift":   lambda g: (g["final_safety_distance"] - g["initial_safety_distance"]).mean(),
    "desc_drift":     lambda g: g["final_description_drift"].mean(),
}

def summarize(df, param_held_constant, param_varying, metric, direction):
    masked = pd.Series(True, index=df.index)
    for col, val in param_held_constant.items():
        masked &= df[col] == val
    masked &= np.isclose(pd.to_numeric(df["direction"], errors="coerce"), float(direction))

    sub = df[masked]
  
    if sub.empty:
        print("no data available.")
        return

    y = sub.groupby(param_varying).apply(METRICS[metric])

    print(f"\n{metric} by {param_varying} (direction={direction}):")
    print(y.to_string())

    plt.figure()
    plt.plot(y.index, y.values, marker = "o")
    plt.xlabel(param_varying)
    plt.ylabel(metric)

    out = Path("graphs")
    out.mkdir(exist_ok=True)
    plt.savefig(out / f"{metric}_vs_{param_varying}_dir{direction}.png", dpi=300, bbox_inches="tight")
    plt.close()

    
    

if __name__== "__main__":

    rows = []
    for p in Path("attack_results").rglob("results_*"):
        if not p.is_file():
            continue
        try:
            with open(p) as f:
                row = {k: v for k, v in json.load(f).items() if not k.startswith("h_")}
            rows.append(row)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    df = pd.DataFrame(rows)

    print(df.columns.tolist())

    default_values = {"epsilon": 1.0, "model_name": "LLaVA-1.5-7b", "pooling_method": "last_token", "layer_from_last": -1}
    for d in [1.0, -1.0]:
        summarize(df, default_values, "mu", "attack_success", d)    