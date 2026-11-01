from __future__ import annotations

import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .splits import build_transfer_splits


if "numpy._core" not in sys.modules and hasattr(np, "core"):
    sys.modules["numpy._core"] = np.core


def prepare_data_index(config: dict[str, Any]) -> dict[str, Any]:
    data_dir = Path(config["data"]["lincs_dir"])
    pathway_dir = Path(config["data"]["pathway_dir"])
    cache_dir = Path(config["data"]["cache_dir"])
    result_dir = Path(config["train"]["result_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    split_file = Path(config["data"].get("split_file", cache_dir / "lincs_splits_v3.json"))
    if not split_file.exists():
        build_transfer_splits(
            data_dir=data_dir,
            cache_dir=cache_dir,
            result_dir=result_dir,
            seed=int(config["train"]["seed"]),
        )

    conditions = pd.read_csv(data_dir / "conditions.csv", usecols=["canonical_smiles", "dose", "time"])
    unique = conditions.drop_duplicates().sort_values(
        ["canonical_smiles", "dose", "time"], kind="mergesort"
    ).reset_index(drop=True)
    unique_index = pd.MultiIndex.from_frame(unique)
    row_to_condition = unique_index.get_indexer(pd.MultiIndex.from_frame(conditions)).astype(np.int32)
    if (row_to_condition < 0).any():
        raise RuntimeError("Failed to map every LINCS row to a molecular condition")
    np.save(cache_dir / "row_to_condition.npy", row_to_condition)
    unique.to_csv(cache_dir / "unique_conditions.csv", index=False)
    smiles_values = sorted(unique["canonical_smiles"].astype(str).unique().tolist())
    smiles_position = {smiles: index for index, smiles in enumerate(smiles_values)}
    condition_to_smiles = unique["canonical_smiles"].astype(str).map(smiles_position).to_numpy(np.int32)
    np.save(cache_dir / "condition_to_smiles.npy", condition_to_smiles)

    x_control = np.load(data_dir / "x_control.npy", mmap_mode="r")
    x_treated = np.load(data_dir / "x_treated.npy", mmap_mode="r")
    reactome = np.load(pathway_dir / "reactome_targets.npy", mmap_mode="r")
    hallmark = np.load(pathway_dir / "hallmark_targets.npy", mmap_mode="r")
    splits = json.loads(split_file.read_text())
    n_rows = len(conditions)
    expected = {
        "x_control": (n_rows, 978),
        "x_treated": (n_rows, 978),
        "reactome": (n_rows, 472),
        "hallmark": (n_rows, 49),
    }
    actual = {
        "x_control": tuple(x_control.shape),
        "x_treated": tuple(x_treated.shape),
        "reactome": tuple(reactome.shape),
        "hallmark": tuple(hallmark.shape),
    }
    if actual != expected:
        raise RuntimeError(f"LINCS shape mismatch: expected={expected}, actual={actual}")
    nonfinite: dict[str, int] = {}
    for name, array in {
        "x_control": x_control,
        "x_treated": x_treated,
        "reactome": reactome,
        "hallmark": hallmark,
    }.items():
        count_nonfinite = 0
        for start in range(0, n_rows, 4096):
            count_nonfinite += int((~np.isfinite(np.asarray(array[start : start + 4096]))).sum())
        nonfinite[name] = count_nonfinite
    if any(nonfinite.values()):
        raise RuntimeError(f"LINCS arrays contain non-finite values: {nonfinite}")

    split_arrays = {name: np.asarray(values, dtype=np.int64) for name, values in splits.items()}
    combined = np.concatenate(list(split_arrays.values()))
    split_ok = len(combined) == n_rows and np.array_equal(np.sort(combined), np.arange(n_rows))
    if not split_ok:
        raise RuntimeError("LINCS train/valid/test indices do not form an exact partition")

    chunk_rows = int(config["data"].get("stats_chunk_rows", 8192))
    train_indices = split_arrays["train"]

    def train_stats(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        total = np.zeros(978, dtype=np.float64)
        total_square = np.zeros(978, dtype=np.float64)
        count = 0
        for start in range(0, len(train_indices), chunk_rows):
            chunk = np.asarray(array[train_indices[start : start + chunk_rows]], dtype=np.float64)
            total += chunk.sum(axis=0)
            total_square += np.square(chunk).sum(axis=0)
            count += len(chunk)
        mean = total / count
        variance = np.maximum(total_square / count - np.square(mean), 1e-8)
        return mean, np.sqrt(variance), count

    control_mean, control_std, count = train_stats(x_control)
    treated_mean, treated_std, treated_count = train_stats(x_treated)
    if treated_count != count:
        raise RuntimeError("Control and treated train statistics used different row counts")
    np.savez(
        cache_dir / "control_stats.npz",
        mean=control_mean.astype(np.float32),
        std=control_std.astype(np.float32),
        treated_mean=treated_mean.astype(np.float32),
        treated_std=treated_std.astype(np.float32),
        count=np.asarray(count, dtype=np.int64),
    )

    genes = [line.strip() for line in (data_dir / "genes.txt").read_text().splitlines() if line.strip()]
    report = {
        "status": "passed",
        "rows": n_rows,
        "genes": len(genes),
        "unique_conditions": len(unique),
        "unique_smiles": int(unique["canonical_smiles"].nunique()),
        "arrays": {name: list(shape) for name, shape in actual.items()},
        "nonfinite": nonfinite,
        "splits": {name: int(len(values)) for name, values in split_arrays.items()},
        "split_file": str(split_file),
        "condition_nulls": {name: int(conditions[name].isna().sum()) for name in conditions.columns},
        "control_stats_finite": bool(np.isfinite(control_mean).all() and np.isfinite(control_std).all()),
        "treated_stats_finite": bool(np.isfinite(treated_mean).all() and np.isfinite(treated_std).all()),
        "molecular_cache_exists": bool((data_dir / "mol_features_2d.pkl").exists()),
    }
    (result_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class LincsRows(Dataset[tuple[int, int]]):
    def __init__(self, config: dict[str, Any], split: str, limit: int | None = None):
        data_dir = Path(config["data"]["lincs_dir"])
        cache_dir = Path(config["data"]["cache_dir"])
        split_file = Path(config["data"].get("split_file", cache_dir / "lincs_splits_v3.json"))
        split_values = json.loads(split_file.read_text())[split]
        self.rows = np.asarray(split_values[:limit] if limit else split_values, dtype=np.int64)
        self.row_to_condition = np.load(cache_dir / "row_to_condition.npy", mmap_mode="r")
        self.conditions = np.asarray(self.row_to_condition[self.rows], dtype=np.int32)
        self.condition_to_smiles = np.load(cache_dir / "condition_to_smiles.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return int(self.rows[index]), int(self.conditions[index])


class ConditionGroupedBatchSampler(Sampler[list[int]]):
    """Group repeated molecular conditions while retaining every split row."""

    def __init__(
        self,
        condition_ids: np.ndarray,
        condition_to_smiles: np.ndarray,
        batch_size: int,
        group_size: int,
        seed: int,
        rank: int,
        world_size: int,
    ):
        self.condition_ids = np.asarray(condition_ids, dtype=np.int64)
        self.condition_to_smiles = np.asarray(condition_to_smiles, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        order = np.argsort(self.condition_ids, kind="stable")
        sorted_conditions = self.condition_ids[order]
        boundaries = np.flatnonzero(np.diff(sorted_conditions)) + 1
        self.groups = np.split(order, boundaries)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.condition_ids) // (self.batch_size * self.world_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        chunks_by_smiles: dict[int, dict[int, list[np.ndarray]]] = {}
        for values in self.groups:
            values = values.copy()
            rng.shuffle(values)
            condition = int(self.condition_ids[values[0]])
            smiles = int(self.condition_to_smiles[condition])
            condition_chunks = np.array_split(values, math.ceil(len(values) / self.group_size))
            chunks_by_smiles.setdefault(smiles, {})[condition] = condition_chunks

        units: list[np.ndarray] = []
        for conditions in chunks_by_smiles.values():
            queues = {condition: list(chunks) for condition, chunks in conditions.items()}
            condition_order = list(queues)
            rng.shuffle(condition_order)
            while sum(bool(queue) for queue in queues.values()) >= 2:
                available = [condition for condition in condition_order if queues[condition]]
                first, second = available[:2]
                units.append(np.concatenate([queues[first].pop(), queues[second].pop()]))
                condition_order = condition_order[2:] + condition_order[:2]
            for queue in queues.values():
                units.extend(queue)
        random.Random(self.seed + self.epoch).shuffle(units)
        sequence = np.concatenate(units)
        usable = (len(sequence) // (self.batch_size * self.world_size)) * self.batch_size * self.world_size
        batches = sequence[:usable].reshape(-1, self.batch_size)
        for batch in batches[self.rank :: self.world_size]:
            yield batch.tolist()


class DistributedSliceSampler(Sampler[int]):
    def __init__(self, size: int, rank: int, world_size: int):
        self.values = range(rank, size, world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def _pad(array: np.ndarray, rows: int, columns: int) -> np.ndarray:
    output = np.zeros((rows, columns), dtype=np.float32)
    output[: array.shape[0], : array.shape[1]] = array
    return output


class LincsCollator:
    def __init__(self, config: dict[str, Any]):
        data_dir = Path(config["data"]["lincs_dir"])
        pathway_dir = Path(config["data"]["pathway_dir"])
        cache_dir = Path(config["data"]["cache_dir"])
        self.x_control = np.load(data_dir / "x_control.npy", mmap_mode="r")
        self.x_treated = np.load(data_dir / "x_treated.npy", mmap_mode="r")
        self.reactome = np.load(pathway_dir / "reactome_targets.npy", mmap_mode="r")
        self.hallmark = np.load(pathway_dir / "hallmark_targets.npy", mmap_mode="r")
        conditions = pd.read_csv(cache_dir / "unique_conditions.csv")
        self.smiles = conditions["canonical_smiles"].astype(str).to_numpy()
        smiles_values = sorted(set(self.smiles.tolist()))
        self.smiles_to_id = {smiles: index for index, smiles in enumerate(smiles_values)}
        self.dose = conditions["dose"].to_numpy(dtype=np.float32) / 100.0
        self.time = conditions["time"].to_numpy(dtype=np.float32) / 72.0
        with (data_dir / "mol_features_2d.pkl").open("rb") as handle:
            self.features = pickle.load(handle)
        missing = sorted(set(self.smiles) - set(self.features))
        if missing:
            raise RuntimeError(f"Read-only molecular cache misses {len(missing)} SMILES")

    def __call__(self, batch: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
        rows = np.asarray([item[0] for item in batch], dtype=np.int64)
        condition_ids = np.asarray([item[1] for item in batch], dtype=np.int64)
        unique_conditions, inverse = np.unique(condition_ids, return_inverse=True)
        condition_smiles = self.smiles[unique_conditions]
        unique_smiles, condition_to_smiles = np.unique(condition_smiles, return_inverse=True)
        graphs = [self.features[smiles] for smiles in unique_smiles]
        max_atoms = max(graph[0].shape[0] for graph in graphs)
        atom_dim = graphs[0][0].shape[1]
        node = np.asarray([_pad(graph[0], max_atoms, atom_dim) for graph in graphs])
        adjacency = np.asarray([_pad(graph[1], max_atoms, max_atoms) for graph in graphs])
        distance = np.asarray([_pad(graph[2], max_atoms, max_atoms) for graph in graphs])
        return {
            "row_index": torch.from_numpy(rows),
            "x_control": torch.from_numpy(np.asarray(self.x_control[rows], dtype=np.float32)),
            "x_treated": torch.from_numpy(np.asarray(self.x_treated[rows], dtype=np.float32)),
            "reactome_target": torch.from_numpy(np.asarray(self.reactome[rows], dtype=np.float32)),
            "hallmark_target": torch.from_numpy(np.asarray(self.hallmark[rows], dtype=np.float32)),
            "node_features": torch.from_numpy(node),
            "mask": torch.from_numpy(np.abs(node).sum(axis=-1) != 0),
            "adjacency": torch.from_numpy(adjacency),
            "distance": torch.from_numpy(distance),
            "dose": torch.from_numpy(self.dose[unique_conditions, None]),
            "time": torch.from_numpy(self.time[unique_conditions, None]),
            "condition_inverse": torch.from_numpy(inverse.astype(np.int64)),
            "condition_to_smiles": torch.from_numpy(condition_to_smiles.astype(np.int64)),
            "unique_smiles_ids": torch.from_numpy(
                np.asarray([self.smiles_to_id[smiles] for smiles in unique_smiles], dtype=np.int64)
            ),
        }


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(
    config: dict[str, Any],
    split: str,
    rank: int,
    world_size: int,
    training: bool,
    limit: int | None = None,
) -> tuple[DataLoader, ConditionGroupedBatchSampler | None]:
    dataset = LincsRows(config, split=split, limit=limit)
    collator = LincsCollator(config)
    workers = int(config["train"]["num_workers_per_gpu"])
    if training:
        batch_sampler = ConditionGroupedBatchSampler(
            dataset.conditions,
            dataset.condition_to_smiles,
            batch_size=int(config["train"]["batch_size_per_gpu"]),
            group_size=int(config["train"]["condition_group_size"]),
            seed=int(config["train"]["seed"]),
            rank=rank,
            world_size=world_size,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            collate_fn=collator,
            worker_init_fn=seed_worker,
        )
        return loader, batch_sampler

    sampler = DistributedSliceSampler(len(dataset), rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"].get("eval_batch_size_per_gpu", config["train"]["batch_size_per_gpu"])),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collator,
        worker_init_fn=seed_worker,
    )
    return loader, None
