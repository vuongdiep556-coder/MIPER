from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _worker() -> None:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    smiles_values = json.loads(sys.stdin.read())
    output: dict[str, str | None] = {}
    for smiles in smiles_values:
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                output[smiles] = None
                continue
            canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=molecule, includeChirality=True
            )
            output[smiles] = scaffold or f"ACYCLIC::{canonical}"
        except Exception:
            output[smiles] = None
    sys.stdout.write(json.dumps(output))


def _scaffold_fallback(smiles: str) -> str:
    digest = hashlib.sha256(smiles.encode()).hexdigest()[:20]
    return f"FAILED::{digest}"


def _extract_scaffold_batch(smiles_values: list[str]) -> tuple[dict[str, str], list[str]]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    result = subprocess.run(
        command,
        input=json.dumps(smiles_values),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        parsed = json.loads(result.stdout)
        failed = [smiles for smiles, scaffold in parsed.items() if scaffold is None]
        return {
            smiles: scaffold if scaffold is not None else _scaffold_fallback(smiles)
            for smiles, scaffold in parsed.items()
        }, failed
    if len(smiles_values) == 1:
        smiles = smiles_values[0]
        return {smiles: _scaffold_fallback(smiles)}, [smiles]
    midpoint = len(smiles_values) // 2
    left, left_failed = _extract_scaffold_batch(smiles_values[:midpoint])
    right, right_failed = _extract_scaffold_batch(smiles_values[midpoint:])
    left.update(right)
    return left, left_failed + right_failed


def extract_scaffolds(smiles_values: list[str], batch_size: int = 256) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    failed: list[str] = []
    for start in range(0, len(smiles_values), batch_size):
        values, failures = _extract_scaffold_batch(smiles_values[start : start + batch_size])
        mapping.update(values)
        failed.extend(failures)
    return mapping, sorted(set(failed))


def _take_groups(
    groups: list[tuple[str, np.ndarray]], target_rows: int, seed: int
) -> tuple[set[str], np.ndarray]:
    rng = random.Random(seed)
    groups = groups.copy()
    rng.shuffle(groups)
    selected_names: set[str] = set()
    selected_rows: list[np.ndarray] = []
    count = 0
    for name, rows in groups:
        if count >= target_rows:
            break
        selected_names.add(name)
        selected_rows.append(rows)
        count += len(rows)
    return selected_names, np.concatenate(selected_rows) if selected_rows else np.empty(0, dtype=np.int64)


def _random_validation_rows(
    available_rows: np.ndarray,
    condition_ids: np.ndarray,
    target_rows: int,
    seed: int,
) -> np.ndarray:
    by_condition: dict[int, list[int]] = defaultdict(list)
    for row in available_rows.tolist():
        by_condition[int(condition_ids[row])].append(int(row))
    remaining = {condition: len(rows) for condition, rows in by_condition.items()}
    candidates = [row for rows in by_condition.values() if len(rows) > 1 for row in rows]
    random.Random(seed).shuffle(candidates)
    selected: list[int] = []
    for row in candidates:
        condition = int(condition_ids[row])
        if remaining[condition] <= 1:
            continue
        selected.append(row)
        remaining[condition] -= 1
        if len(selected) >= target_rows:
            break
    if len(selected) < target_rows:
        raise RuntimeError(
            f"Could only allocate {len(selected):,}/{target_rows:,} random validation rows "
            "while retaining every condition in train"
        )
    return np.asarray(selected, dtype=np.int64)


def build_transfer_splits(
    data_dir: Path,
    cache_dir: Path,
    result_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    conditions = pd.read_csv(
        data_dir / "conditions.csv", usecols=["canonical_smiles", "dose", "time"]
    )
    if conditions.isna().any().any():
        raise RuntimeError("LINCS condition fields contain missing values")
    conditions["canonical_smiles"] = conditions["canonical_smiles"].astype(str)
    original = json.loads((data_dir / "splits.json").read_text())
    original_arrays = {
        name: np.asarray(original[name], dtype=np.int64) for name in ("train", "valid", "test")
    }
    n_rows = len(conditions)
    original_partition = np.concatenate(list(original_arrays.values()))
    if len(original_partition) != n_rows or not np.array_equal(
        np.sort(original_partition), np.arange(n_rows)
    ):
        raise RuntimeError("Original LINCS split is not a complete row partition")

    unique_conditions = conditions.drop_duplicates().sort_values(
        ["canonical_smiles", "dose", "time"], kind="mergesort"
    ).reset_index(drop=True)
    condition_index = pd.MultiIndex.from_frame(unique_conditions)
    row_to_condition = condition_index.get_indexer(pd.MultiIndex.from_frame(conditions)).astype(np.int32)

    smiles_values = sorted(conditions["canonical_smiles"].unique().tolist())
    scaffold_map, failed_smiles = extract_scaffolds(smiles_values)
    scaffold_frame = pd.DataFrame(
        {"canonical_smiles": smiles_values, "scaffold": [scaffold_map[s] for s in smiles_values]}
    )
    scaffold_frame.to_csv(cache_dir / "scaffolds_v3.csv", index=False)
    row_smiles = conditions["canonical_smiles"].to_numpy()
    row_scaffold = np.asarray([scaffold_map[value] for value in row_smiles], dtype=object)

    pool = np.concatenate([original_arrays["train"], original_arrays["valid"]])
    target_ood = int(round(n_rows * 0.05))
    target_random = len(original_arrays["valid"])

    pool_frame = pd.DataFrame(
        {
            "row": pool,
            "smiles": row_smiles[pool],
            "scaffold": row_scaffold[pool],
        }
    )
    scaffold_drugs = pool_frame.groupby("scaffold")["smiles"].nunique()
    multi_scaffolds = set(scaffold_drugs.index[scaffold_drugs >= 2].astype(str))
    scaffold_groups = [
        (str(scaffold), group["row"].to_numpy(np.int64))
        for scaffold, group in pool_frame.groupby("scaffold", sort=False)
        if str(scaffold) in multi_scaffolds
    ]
    scaffold_names, scaffold_rows = _take_groups(scaffold_groups, target_ood, seed + 101)
    scaffold_mask = np.isin(pool, scaffold_rows)
    after_scaffold = pool[~scaffold_mask]

    remaining_frame = pd.DataFrame(
        {
            "row": after_scaffold,
            "smiles": row_smiles[after_scaffold],
            "scaffold": row_scaffold[after_scaffold],
        }
    )
    remaining_scaffold_drugs = remaining_frame.groupby("scaffold")["smiles"].nunique()
    eligible_scaffolds = set(remaining_scaffold_drugs.index[remaining_scaffold_drugs >= 2].astype(str))
    drug_groups = []
    for smiles, group in remaining_frame.groupby("smiles", sort=False):
        scaffold = str(group["scaffold"].iloc[0])
        if scaffold in eligible_scaffolds:
            drug_groups.append((str(smiles), group["row"].to_numpy(np.int64)))
    drug_names, provisional_drug_rows = _take_groups(drug_groups, target_ood, seed + 202)

    # Keep at least one sibling drug from every selected drug's scaffold in train.
    selected_drugs_by_scaffold: dict[str, list[str]] = defaultdict(list)
    for smiles in drug_names:
        selected_drugs_by_scaffold[scaffold_map[smiles]].append(smiles)
    for scaffold, selected in selected_drugs_by_scaffold.items():
        available = sorted(
            set(remaining_frame.loc[remaining_frame["scaffold"] == scaffold, "smiles"].astype(str))
        )
        if len(selected) >= len(available):
            drug_names.remove(sorted(selected)[-1])
    drug_rows = remaining_frame.loc[
        remaining_frame["smiles"].isin(drug_names), "row"
    ].to_numpy(np.int64)
    after_drug = after_scaffold[~np.isin(after_scaffold, drug_rows)]

    random_rows = _random_validation_rows(
        after_drug, row_to_condition, target_random, seed + 303
    )
    train_rows = after_drug[~np.isin(after_drug, random_rows)]
    splits = {
        "train": np.sort(train_rows),
        "random_valid": np.sort(random_rows),
        "drug_ood_valid": np.sort(drug_rows),
        "scaffold_ood_valid": np.sort(scaffold_rows),
        "test": np.sort(original_arrays["test"]),
    }

    combined = np.concatenate(list(splits.values()))
    if len(combined) != n_rows or not np.array_equal(np.sort(combined), np.arange(n_rows)):
        raise RuntimeError("Transfer-aware LINCS splits are not an exact row partition")
    if len(np.unique(combined)) != n_rows:
        raise RuntimeError("Transfer-aware LINCS splits overlap")

    train_drugs = set(row_smiles[splits["train"]])
    random_drugs = set(row_smiles[splits["random_valid"]])
    drug_ood_drugs = set(row_smiles[splits["drug_ood_valid"]])
    scaffold_ood_drugs = set(row_smiles[splits["scaffold_ood_valid"]])
    train_scaffolds = set(row_scaffold[splits["train"]])
    drug_ood_scaffolds = set(row_scaffold[splits["drug_ood_valid"]])
    scaffold_ood_scaffolds = set(row_scaffold[splits["scaffold_ood_valid"]])
    train_conditions = set(row_to_condition[splits["train"]].tolist())
    random_conditions = set(row_to_condition[splits["random_valid"]].tolist())
    checks = {
        "random_drugs_seen": random_drugs <= train_drugs,
        "random_conditions_seen": random_conditions <= train_conditions,
        "drug_ood_drugs_disjoint": not bool(drug_ood_drugs & train_drugs),
        "drug_ood_scaffolds_seen": drug_ood_scaffolds <= train_scaffolds,
        "scaffold_ood_drugs_disjoint": not bool(scaffold_ood_drugs & train_drugs),
        "scaffold_ood_scaffolds_disjoint": not bool(scaffold_ood_scaffolds & train_scaffolds),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Transfer split leakage audit failed: {checks}")

    output_path = cache_dir / "lincs_splits_v3.json"
    output_path.write_text(
        json.dumps({name: rows.tolist() for name, rows in splits.items()}, separators=(",", ":"))
    )
    report = {
        "status": "passed",
        "seed": seed,
        "rows": n_rows,
        "drugs": len(smiles_values),
        "scaffolds": int(scaffold_frame["scaffold"].nunique()),
        "failed_scaffold_smiles": failed_smiles,
        "splits": {
            name: {
                "rows": int(len(rows)),
                "drugs": int(len(set(row_smiles[rows]))),
                "scaffolds": int(len(set(row_scaffold[rows]))),
            }
            for name, rows in splits.items()
        },
        "targets": {"ood_rows": target_ood, "random_valid_rows": target_random},
        "checks": checks,
        "split_file": str(output_path),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "transfer_split_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--data-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--result-dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.worker:
        _worker()
        return
    if not args.data_dir or not args.cache_dir or not args.result_dir:
        parser.error("--data-dir, --cache-dir, and --result-dir are required")
    report = build_transfer_splits(
        Path(args.data_dir), Path(args.cache_dir), Path(args.result_dir), args.seed
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
