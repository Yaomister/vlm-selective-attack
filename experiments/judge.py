import os
import json
import ollama
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from parse import parse_safety_response
from utils import sweep_result_file_pattern


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--results-dir", default="./attack_results")
    parser.add_argument("--output-dir", default="./judged_results")

    return parser.parse_args()


def judge(before, after):
    prompt = (
        "Do these describe the same main subject and scene? Minor wording differences are fine. "
        "Answer with exactly one word: yes or no.\n\n"
        f"Description 1: {before}\n\nDescription 2: {after}"
    )
    r = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return r["message"]["content"].strip()


def load_results(results_dir):
    path = Path(results_dir)

    data = []

    for file in path.rglob("*.json"):
        match = sweep_result_file_pattern.match(os.path.basename(file))
        if match:
            experiment_details = match.groupdict()
            with open(file, encoding="utf-8") as f:
                results = json.load(f)

            data.append({**experiment_details, **results})

    df = pd.DataFrame(data)
    df.groupby(["model"])
    df = df[df["epsilon"] == 0.025]

    return df


if __name__ == "__main__":
    args = parse_arguments()

    print("Starting...")

    data = load_results(args.results_dir)

    print("Data loaded successfully, starting to judge the responses.")

    res = []

    for model_name in data["model"].unique():
        for attack_direction in data["direction"].unique():

            print(f"Starting for {model_name} direction {attack_direction}")

            subset = data[
                (data["model"] == model_name) & (data["direction"] == attack_direction)
            ]

            total_judged = 0
            amount_said_yes = 0

            for response_before, response_after in zip(
                subset["description_clean"],
                subset["description_perturbed"],
            ):

                judged_response = judge(response_before, response_after)

                total_judged += 1
                if parse_safety_response(judged_response).lower() == "yes":
                    amount_said_yes += 1
                else:
                    print("Skipped")
                    print(judged_response)
                    print(response_before)
                    print(response_after)

                res.append(
                    {
                        "model": model_name,
                        "direction": attack_direction,
                        "judged_response": judged_response,
                    }
                )
                print(f"Done {total_judged}")

            asr = float(amount_said_yes / total_judged)

            res.append({"ASR": asr})

            output_path = Path(
                f"{args.output_dir}/judged_responses_{model_name}_{dir}.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(res, f)

            print(f"for {model_name} in the direction {attack_direction}: {asr:.4f}")

    print("Finished judgeing the responses.")
