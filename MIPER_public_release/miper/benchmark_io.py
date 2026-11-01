from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset, Subset

from .chem import pad_array

DATASETS = ("TGSA", "GDSC1", "GDSC2", "CCLE", "CTRP1", "CTRP2")

class BenchmarkDataset(Dataset):
    """Compact cell-level expression and packed molecular graph dataset."""

    def __init__(self, cfg: dict[str, Any], dataset: str):
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset: {dataset}")
        data_root = Path(cfg["data"]["root"])
        processed = data_root / "datasets" / dataset
        pairs = np.load(processed / "pairs.npz", allow_pickle=False)
        self.row_id = pairs["row_id"].astype(np.int64, copy=False)
        self.cell_index = pairs["cell_index"].astype(np.int32, copy=False)
        self.graph_index = pairs["graph_index"].astype(np.int32, copy=False)
        self.y = pairs["y"].astype(np.float32, copy=False)
        self.cell_gex = np.load(processed / "cell_gex.npy", mmap_mode="r")

        graph_root = data_root / "graphs_2d"
        graph_frame = pd.read_csv(graph_root / "graph_index.tsv", sep="\t").sort_values(
            "graph_index"
        )
        expected = np.arange(len(graph_frame), dtype=np.int64)
        if not np.array_equal(graph_frame["graph_index"].to_numpy(np.int64), expected):
            raise RuntimeError("graph_index.tsv must be contiguous and sorted")
        self.atom_count = graph_frame["atom_count"].to_numpy(np.int32)
        self.node_offset = graph_frame["node_offset"].to_numpy(np.int64)
        self.square_offset = graph_frame["square_offset"].to_numpy(np.int64)
        self.nodes = np.load(graph_root / "node_features.npy", mmap_mode="r")
        self.adjacency = np.load(graph_root / "adjacency.npy", mmap_mode="r")
        self.distance = np.load(graph_root / "distance.npy", mmap_mode="r")
        self.dose = np.float32(cfg["train"]["scaled_dose"])
        self.time = np.float32(cfg["train"]["time"])

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, index: int) -> dict[str, Any]:
        graph = int(self.graph_index[index])
        atoms = int(self.atom_count[graph])
        node_offset = int(self.node_offset[graph])
        square_offset = int(self.square_offset[graph])
        return {
            "row_index": int(self.row_id[index]),
            "x_basal": np.asarray(
                self.cell_gex[int(self.cell_index[index])], dtype=np.float32
            ).copy(),
            "ic50": np.float32(self.y[index]),
            "node_features": np.asarray(
                self.nodes[node_offset : node_offset + atoms], dtype=np.float32
            ).copy(),
            "adjacency": np.asarray(
                self.adjacency[square_offset : square_offset + atoms * atoms],
                dtype=np.float32,
            ).reshape(atoms, atoms).copy(),
            "distance": np.asarray(
                self.distance[square_offset : square_offset + atoms * atoms],
                dtype=np.float32,
            ).reshape(atoms, atoms).copy(),
            "dose": self.dose,
            "time": self.time,
        }


def split_seed(cfg: dict[str, Any]) -> int:
    configured = cfg["data"].get("split_seed", 41)
    return int(cfg["train"]["seed"]) if configured == "train_seed" else int(configured)


def split_partitions(cfg: dict[str, Any]) -> tuple[str, ...]:
    return tuple(cfg["data"].get("split_partitions", ("train", "valid", "test", "unused")))


def split_path(cfg: dict[str, Any], dataset: str, partition: str) -> Path:
    split_root = Path(cfg["data"].get("split_root", Path(cfg["data"]["root"]) / "splits"))
    return split_root / dataset / f"seed{split_seed(cfg)}" / f"{partition}.npy"


def split_indices(cfg: dict[str, Any], dataset: str, partition: str) -> np.ndarray:
    path = split_path(cfg, dataset, partition)
    return np.load(path, allow_pickle=False).astype(np.int64, copy=False)


def make_loader(
    dataset: BenchmarkDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_gdsc,
    )


def metric_values(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - y
    values: dict[str, float | int] = {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pcc": float(np.corrcoef(y, prediction)[0, 1]),
        "mae": float(np.mean(np.abs(error))),
        "scc": float(spearmanr(y, prediction).statistic),
        "mse": float(np.mean(np.square(error))),
    }
    if not all(math.isfinite(float(value)) for key, value in values.items() if key != "n"):
        raise RuntimeError("non-finite evaluation metric")
    return values


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        mask = batch["node_features"].abs().sum(dim=-1).ne(0)
        output = model(
            batch["x_basal"],
            batch["node_features"],
            mask,
            batch["adjacency"],
            batch["distance"],
            batch["dose"],
            batch["time"],
        )
        rows.append(batch["row_index"].cpu().numpy())
        labels.append(batch["ic50"].cpu().numpy())
        predictions.append(output["pred_ic50"].cpu().numpy().reshape(-1))
    return np.concatenate(rows), np.concatenate(labels), np.concatenate(predictions)


def collate_gdsc(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    """Pad molecular graphs and stack GDSC regression examples."""
    out = {
        "row_index": torch.from_numpy(np.asarray([item["row_index"] for item in batch], dtype=np.int64)),
        "x_basal": torch.from_numpy(np.stack([item["x_basal"] for item in batch])),
        "ic50": torch.as_tensor([item["ic50"] for item in batch], dtype=torch.float32),
        "dose": torch.as_tensor([item["dose"] for item in batch], dtype=torch.float32).unsqueeze(-1),
        "time": torch.as_tensor([item["time"] for item in batch], dtype=torch.float32).unsqueeze(-1),
    }
    if "compound_embedding" in batch[0]:
        out["compound_embedding"] = torch.from_numpy(np.stack([item["compound_embedding"] for item in batch]))
    else:
        max_atoms = int(max(int(item["node_features"].shape[0]) for item in batch))  # type: ignore[index, union-attr]
        feature_dim = int(batch[0]["node_features"].shape[1])  # type: ignore[index, union-attr]
        node_features, adjacency, distance = [], [], []
        for item in batch:
            node_features.append(pad_array(item["node_features"], (max_atoms, feature_dim)))  # type: ignore[arg-type]
            adjacency.append(pad_array(item["adjacency"], (max_atoms, max_atoms)))  # type: ignore[arg-type]
            distance.append(pad_array(item["distance"], (max_atoms, max_atoms)))  # type: ignore[arg-type]
        out.update(
            {
                "adjacency": torch.from_numpy(np.stack(adjacency)),
                "node_features": torch.from_numpy(np.stack(node_features)),
                "distance": torch.from_numpy(np.stack(distance)),
            }
        )
    return out
