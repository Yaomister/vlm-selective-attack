import re
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import cycle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import sweep_result_file_pattern
from parse import parse_safety_response

# make it print everything
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


okabe_ito_palette = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#000000",
]

graph_markers = ["o", "s", "^", "D", "v", "*", "X"]


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", default="./attack_results")
    parser.add_argument("--graph-output-dir", default="./graphs")
    parser.add_argument(
        "--what-to-analyse",
        choices=[
            "epsilon-sweep",
            "mu-sweep",
            "geometric",
            "ablation",
            "geometric-baseline",
        ],
        required=True,
    )
    return parser.parse_args()


def load_ablation_dataset(data_dir):
    rows = []

    for file in Path(data_dir).rglob("*.json"):
        content = json.loads(file.read_text())

        for tested in content.values():
            for epsilon in tested.values():
                for image in epsilon.values():
                    for image_path, records in image.items():
                        for record in records:
                            rows.append(
                                {"ablation": file.stem, "image": image_path, **record}
                            )

    df = pd.DataFrame(rows)
    df["epsilon"] = pd.to_numeric(df["epsilon"])
    return df


def load_sweeps_dataset(data_dir):
    data = []
    root = Path(data_dir)

    numeric_columns = [
        "layer_from_last",
        "mu",
        "epsilon",
        "direction",
        "pair_id",
        "pgd_steps",
    ]

    for file in root.rglob("*.json"):
        if not file.is_file():
            continue

        match = sweep_result_file_pattern.match(os.path.basename(file))

        if match:
            experiment_details = match.groupdict()
            content = json.loads(file.read_text())
            data.append(
                {
                    **experiment_details,
                    **content,
                }
            )

    df = pd.DataFrame(data)
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    return df


def _before_flip(data):
    clean = data["safety_clean"].map(parse_safety_response)
    harmful = data["direction"].astype(float) > 0
    return (harmful & (clean == "no")) | (~harmful & (clean == "yes"))


def _successful_flips(data):
    pert = data["safety_perturbed"].map(parse_safety_response)
    harmful = data["direction"].astype(float) > 0
    flipped = (harmful & (pert == "yes")) | (~harmful & (pert == "no"))
    return _before_flip(data) & flipped


def _get_marker_color_per_model(model_names):
    pairs = {}

    for model_name, color, marker in zip(model_names, okabe_ito_palette, graph_markers):
        pairs[model_name] = (color, marker)

    return pairs


def calculate_success_rate_over_ablations(data: pd.DataFrame):
    # safety prompt with description
    answer = data["response"].map(parse_safety_response)
    target = data["attack_direction"].map(
        {"harmful_to_safe": "yes", "safe_to_harmful": "no"}
    )

    data = data.assign(success=answer == target)
    success_rates = (
        data.groupby(["ablation", "attack_direction", "trained_model", "tested_model"])[
            "success"
        ]
        .mean()
        .unstack("tested_model")
    )

    print(success_rates)


def calculate_and_graph_success_rate_over_sweep(
    data: pd.DataFrame, output_dir: str, metric: str
):
    data.groupby([metric, "model_name"])

    data = data.assign(eligible=_before_flip(data), success=_successful_flips(data))

    print(f"There are {len(data)} attacks ran")

    success_rates = (
        data.groupby([metric, "model", "direction"])["success"].mean().unstack("model")
    )

    print("The success rates (grouped by model):")
    print(success_rates)

    # names = {1.0: ("harmful to safe", "-"), -1.0: ("safe to harmful", "--")}

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, layout="constrained")
    color_and_marker = _get_marker_color_per_model(success_rates.columns.tolist())

    for ax, (direction, row) in zip(axes, success_rates.groupby("direction")):

        row = row.droplevel("direction")
        for model in row.columns:
            color, marker = color_and_marker[model]
            ax.plot(row.index, row[model], marker=marker, label=model, color=color)
        ax.set_title("Harmful to Safe" if direction == 1 else "Safe to Harmful")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(row.index, labels=[f"{e:g}" for e in row.index], rotation=45)

        axes[0].set_ylim(0, 1)
        axes[-1].legend(loc="lower right")
        fig.supxlabel("ε")
        fig.supylabel("Attack Success Rate")

        # for direction, row in success_rates.groupby("direction"):

        #     fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharey=True)
        #     axes = axes.flatten()
        #     fig.subplots_adjust(hspace=0.35)

        #     row = row.droplevel("direction")
        #     for ax, model in zip(axes, row.columns):
        #         ax.plot(row.index, row[model], marker="o")
        #         ax.set_title(model)
        #         ax.grid(True, alpha=0.3)
        #         ax.set_xscale("log")
        #         ax.minorticks_off()
        #         ax.set_xticks(row.index, labels=[f"{e:g}" for e in row.index], rotation=45)

        #     axes[0].set_ylim(0, 1)
        #     fig.supxlabel("ε")
        #     fig.supylabel("Attack Success Rate")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    fig.savefig(
        output_path / f"success_vs_{metric}.pdf",
        bbox_inches="tight",
    )


def calculate_average_geometric_values(data: pd.DataFrame):
    data = data.assign(eligible=_before_flip(data), success=_successful_flips(data))
    data["final_safety_drift"] = (
        data["final_safety_distance"] - data["initial_safety_distance"]
    )

    data = data[data["eligible"]]

    averages = data.groupby(["success", "direction", "model", "epsilon"])[
        [
            "cos_safety",
            "cos_desc",
            "delta_l2",
            "final_safety_drift",
            "final_description_drift",
        ]
    ].mean()
    print(averages)
    return averages


def calculate_baseline(data: pd.DataFrame):

    data = data.assign(eligible=_before_flip(data), success=_successful_flips(data))

    print("Calculating the normalized safety dift")

    for model, model_data in data.groupby("model"):
        print(f"For model {model}")

        # the normalized safety drift
        safe_mask = (model_data["direction"] < 0).to_numpy()

        safe_centroid = (
            np.stack(model_data[safe_mask]["h_clean_safety"]).astype(np.float64).mean(0)
        )
        harmful_centroid = (
            np.stack(model_data[~safe_mask]["h_clean_safety"])
            .astype(np.float64)
            .mean(0)
        )
        clean_s = np.stack(model_data["h_clean_safety"]).astype(np.float64)
        pert_s = np.stack(model_data["h_pert_safety"]).astype(np.float64)

        model_data["safety_drift"] = ((pert_s - safe_centroid) ** 2).mean(1) - (
            (clean_s - safe_centroid) ** 2
        ).mean(1)

        gap = ((safe_centroid - harmful_centroid) ** 2).mean()

        model_data["normalized_safety_drift"] = model_data["safety_drift"] / gap

        print(
            model_data.groupby(["direction", "success"])[
                "normalized_safety_drift"
            ].mean()
        )

    print("Calculating the normalized description drift")

    for model, model_data in data.groupby("model"):
        print(f"For model {model}")

        pairs = model_data.drop_duplicates(["pair_id", "direction"]).pivot(
            index="pair_id", columns="direction", values="h_clean_desc"
        )

        other = [
            pairs.loc[pair_id, -direction]
            for pair_id, direction in zip(
                model_data["pair_id"], model_data["direction"]
            )
        ]

        other = np.stack(other)

        clean_d = np.stack(model_data["h_clean_desc"]).astype(np.float64)
        pert_d = np.stack(model_data["h_pert_desc"]).astype(np.float64)

        drift = ((clean_d - pert_d) ** 2).mean(1)
        total_distance = ((clean_d - other) ** 2).mean(1)
        model_data["normalized_desc_drift"] = drift / total_distance

        print(
            model_data.groupby(["direction", "success"])["normalized_desc_drift"].mean()
        )

    pairs = model_data.drop_duplicates(["model", "pair_id", "direction"]).pivot(
        index=["model", "pair_id"], columns="direction", values="h_clean_desc"
    )

    # the cosine similarity baseline
    for model, pair in pairs.groupby("model"):
        safe = np.stack(pair[-1.0])
        harmful = np.stack(pair[1.0])
        cosine_partner = (safe * harmful).sum(1) / (
            np.linalg.norm(safe, axis=1) * np.linalg.norm(harmful, axis=1)
        )
        print(model, round(cosine_partner.mean(), 4), round(cosine_partner.std(), 4))


if __name__ == "__main__":
    args = parse_arguments()

    if args.what_to_analyse == "ablation":
        data = load_ablation_dataset(args.data_dir)
    else:
        data = load_sweeps_dataset(args.data_dir)

    match args.what_to_analyse:
        case "geometric-baseline":
            calculate_baseline(data)
        case "epsilon-sweep":
            calculate_and_graph_success_rate_over_sweep(
                data, args.graph_output_dir, "epsilon"
            )
        case "mu-sweep":
            calculate_and_graph_success_rate_over_sweep(
                data, args.graph_output_dir, "mu"
            )
        case "geometric":
            calculate_average_geometric_values(data)
        case "ablation":
            calculate_success_rate_over_ablations(data)
