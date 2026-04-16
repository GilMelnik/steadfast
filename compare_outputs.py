from pathlib import Path
import json
import csv
import re


def export_eval_results_to_csv(root_path: str, output_csv: str | None = None) -> None:
    root = Path(root_path)
    if output_csv is None:
        output_csv = f"{root_path}/eval_results.csv"

    # Matches: 2026_04_15_21_14_on_real_test
    folder_pattern = re.compile(
        r"^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_"
        r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<change>.+)$"
    )

    rows = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue

        match = folder_pattern.match(folder.name)
        if not match:
            print(f"Skipping folder with unexpected name format: {folder.name}")
            continue

        eval_file = folder / "eval_results.json"
        if not eval_file.exists():
            print(f"Missing eval_results.json in: {folder.name}")
            continue

        with eval_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", {})

        row = {
            "date": f"{match.group('year')}-{match.group('month')}-{match.group('day')}",
            "time": f"{match.group('hour')}:{match.group('minute')}",
            "change": match.group("change"),
            "num_tickets": metrics.get("num_tickets"),
            "category_accuracy_exact": metrics.get("category_accuracy_exact"),
            "priority_accuracy_exact": metrics.get("priority_accuracy_exact"),
            "response_quality_score": metrics.get("response_quality_score"),
        }

        rows.append(row)

    fieldnames = [
        "date",
        "time",
        "change",
        "num_tickets",
        "category_accuracy_exact",
        "priority_accuracy_exact",
        "response_quality_score",
    ]

    with Path(output_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


# Example usage
if __name__ == "__main__":
    export_eval_results_to_csv("./outputs")