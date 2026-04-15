import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split


def _resolve_output_paths(
    input_file: Union[str, Path],
    train_file: Optional[Union[str, Path]],
    test_file: Optional[Union[str, Path]],
) -> Tuple[Path, Path]:
    input_path = Path(input_file)
    output_dir = input_path.parent
    resolved_train = Path(train_file) if train_file is not None else output_dir / "train.csv"
    resolved_test = Path(test_file) if test_file is not None else output_dir / "test.json"
    return resolved_train, resolved_test


def _build_stratify_series(dataframe: pd.DataFrame, columns: List[str]) -> pd.Series:
    return dataframe[columns].astype(str).agg("-".join, axis=1)


def _is_valid_stratify_series(stratify_series: pd.Series, test_rows: int) -> bool:
    class_counts = stratify_series.value_counts(dropna=False)
    if class_counts.empty:
        return False

    return class_counts.min() >= 2 and len(class_counts) <= test_rows


def find_stratify_columns(
    dataframe: pd.DataFrame,
    split_ratio: float,
    max_unique_ratio: float = 0.2,
) -> List[str]:
    """Find recurring fields that are suitable for representative stratified sampling."""
    if dataframe.empty:
        return []

    row_count = len(dataframe)
    test_rows = max(1, int(round(row_count * (1 - split_ratio))))
    candidate_columns: List[Tuple[str, int]] = []

    for column in dataframe.columns:
        column_data = dataframe.loc[:, column]
        if isinstance(column_data, pd.DataFrame):
            # If a header is duplicated, merge duplicate columns to evaluate recurrence.
            column_series = column_data.astype(str).agg("|".join, axis=1)
        else:
            column_series = column_data

        unique_count = column_series.nunique(dropna=False)
        if unique_count < 2:
            continue

        unique_ratio = unique_count / row_count
        if unique_ratio <= max_unique_ratio:
            candidate_columns.append((column, unique_count))

    candidate_columns.sort(key=lambda item: item[1])
    selected_columns: List[str] = []
    for column, _ in candidate_columns:
        proposed_columns = selected_columns + [column]
        stratify_series = _build_stratify_series(dataframe, proposed_columns)
        if _is_valid_stratify_series(stratify_series, test_rows):
            selected_columns = proposed_columns

    return selected_columns


def split_csv(
    input_file: Union[str, Path],
    train_file: Optional[Union[str, Path]] = None,
    test_file: Optional[Union[str, Path]] = None,
    split_ratio: float = 0.8,
    num_samples: Optional[int] = None,
    random_seed: int = 42,
    stratify_columns: Optional[List[str]] = None,
) -> None:
    """
    Splits a CSV file into training and testing sets with optional stratification.

    Args:
        input_file: Path to the input CSV file.
        train_file: Path to save the training set.
            If None, saves to "train.csv" next to input_file.
        test_file: Path to save the testing set.
            If None, saves to "test.json" next to input_file.
        split_ratio: Proportion of data for training (default 0.8). Used if num_samples is None.
        num_samples: Fixed number of samples for the test set. Overrides split_ratio if provided.
        random_seed: Random seed for reproducibility.
        stratify_columns: List of column names to stratify on for representative sampling.
            If None, defaults to ["category", "priority"] if they exist, otherwise auto-detected.
    """
    df = pd.read_csv(input_file)
    resolved_train_file, resolved_test_file = _resolve_output_paths(
        input_file=input_file,
        train_file=train_file,
        test_file=test_file,
    )

    # Determine test size
    if num_samples is not None:
        test_size = min(num_samples, len(df) - 1)
    else:
        test_size = 1 - split_ratio

    # Determine stratification columns
    if stratify_columns is None:
        # Try to use category and priority as requested, either combined or separately
        possible_stratify_cols = [
            ["category", "priority"],
            ["category"],
            ["priority"]
        ]
        # Calculate test rows for validation
        test_rows = test_size if isinstance(test_size, int) else max(1, int(round(len(df) * test_size)))

        for group in possible_stratify_cols:
            if all(col in df.columns for col in group):
                s_series = _build_stratify_series(df, group)
                if _is_valid_stratify_series(s_series, test_rows):
                    stratify_columns = group
                    break

        if stratify_columns is None:
            # Fallback to auto-detection
            effective_split_ratio = split_ratio if num_samples is None else 1 - (test_size / len(df))
            stratify_columns = find_stratify_columns(df, split_ratio=effective_split_ratio)

    if stratify_columns:
        stratify_series = _build_stratify_series(df, stratify_columns)
        # Final validation of stratification series against actual test size
        test_rows = test_size if isinstance(test_size, int) else max(1, int(round(len(df) * test_size)))
        stratify = stratify_series if _is_valid_stratify_series(stratify_series, test_rows) else None
    else:
        stratify = None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify
    )

    # Format test_df to match eval_set.json
    test_df = test_df.rename(columns={
        "category": "expected_category",
        "priority": "expected_priority"
    })

    # Select only required columns for test_df
    test_df = test_df[[
        "ticket_id", "customer_name", "plan", "subject",
        "body", "expected_category", "expected_priority"
    ]]

    train_df.to_csv(resolved_train_file, index=False)

    # Save test_df to JSON with indentation
    test_records = test_df.to_dict(orient="records")
    with resolved_test_file.open("w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)


if __name__ == "__main__":
    split_csv(
        input_file="data/knowledge_base.csv",
        num_samples=35,
        stratify_columns=["category", "priority"],
    )
