from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import pandas as pd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

RELEASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from .model import MIPER, parameter_summary
from .pretrain_data import ConditionGroupedBatchSampler, make_loader, prepare_data_index


LOSS_FIELDS = [
    "loss",
    "gene_loss",
    "responsive_delta_loss",
    "delta_pcc_loss",
    "reactome_loss",
    "hallmark_loss",
    "embedding_variance_loss",
    "embedding_covariance_loss",
    "molecular_variance_loss",
    "molecular_covariance_loss",
    "identity_consistency_loss",
]
METRIC_FIELDS = [
    "gene_mse",
    "gene_rmse",
    "gene_pcc",
    "delta_gene_mse",
    "delta_gene_rmse",
    "delta_gene_pcc",
    "reactome_mse",
    "reactome_rmse",
    "reactome_pcc",
    "hallmark_mse",
    "hallmark_rmse",
    "hallmark_pcc",
]
ROW_FIELDS = [
    "epoch",
    "split",
    *LOSS_FIELDS,
    *METRIC_FIELDS,
    "score",
    "lr",
    "seconds",
    "samples_per_second",
    "condition_reuse",
    "smiles_reuse",
    "identity_pair_fraction",
    "embedding_effective_rank",
    "embedding_participation_rank",
    "embedding_top10_fraction",
    "attention_content",
    "attention_bond",
    "attention_distance",
]


def resolve_config(value: Any, project: str) -> Any:
    if isinstance(value, str):
        return (
            value.replace("${PROJECT_ROOT}", project)
            .replace("${RELEASE_ROOT}", str(RELEASE_ROOT))
            .replace("${project}", project)
        )
    if isinstance(value, dict):
        return {key: resolve_config(item, project) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_config(item, project) for item in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    configured = str(config.get("project", PROJECT_ROOT))
    project = os.environ.get("MIPER_PROJECT_ROOT", str(PROJECT_ROOT)) if configured == "${PROJECT_ROOT}" else configured
    return resolve_config(config, project)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def setup_distributed(seed: int) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        if device.type != "cuda":
            raise RuntimeError("Distributed MIPER training requires CUDA")
        dist.init_process_group("nccl")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    return rank, local_rank, world_size, device


def distributed_barrier() -> None:
    if not dist.is_initialized():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def finish_distributed() -> None:
    if dist.is_initialized():
        distributed_barrier()
        dist.destroy_process_group()


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


class RegressionStats:
    def __init__(self, device: torch.device):
        self.value = torch.zeros(7, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, target: torch.Tensor, prediction: torch.Tensor) -> None:
        target = target.detach().double().flatten()
        prediction = prediction.detach().double().flatten()
        difference = prediction - target
        self.value += torch.stack(
            [
                target.new_tensor(target.numel()),
                torch.dot(difference, difference),
                target.sum(),
                prediction.sum(),
                torch.dot(target, target),
                torch.dot(prediction, prediction),
                torch.dot(target, prediction),
            ]
        )

    def compute(self) -> dict[str, float]:
        n, sse, sum_y, sum_p, sum_yy, sum_pp, sum_yp = reduce_sum(self.value.clone()).cpu().tolist()
        mse = sse / max(n, 1.0)
        numerator = n * sum_yp - sum_y * sum_p
        denominator = math.sqrt(max((n * sum_yy - sum_y**2) * (n * sum_pp - sum_p**2), 1e-12))
        return {"mse": mse, "rmse": math.sqrt(max(mse, 0.0)), "pcc": numerator / denominator}


class EpochStats:
    def __init__(self, device: torch.device, track_embedding_rank: bool):
        self.device = device
        self.losses = torch.zeros(len(LOSS_FIELDS), dtype=torch.float64, device=device)
        self.samples = torch.zeros(1, dtype=torch.float64, device=device)
        self.unique_conditions = torch.zeros(1, dtype=torch.float64, device=device)
        self.unique_smiles = torch.zeros(1, dtype=torch.float64, device=device)
        self.paired_conditions = torch.zeros(1, dtype=torch.float64, device=device)
        self.track_embedding_rank = bool(track_embedding_rank)
        self.embedding_by_id: dict[int, tuple[np.ndarray, int]] = {}
        self.regression = {
            "gene": RegressionStats(device),
            "delta_gene": RegressionStats(device),
            "reactome": RegressionStats(device),
            "hallmark": RegressionStats(device),
        }

    @torch.no_grad()
    def update(
        self,
        batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor | None],
        auxiliary: dict[str, torch.Tensor],
        losses: dict[str, torch.Tensor],
    ) -> None:
        batch_size = batch["x_control"].shape[0]
        for index, name in enumerate(LOSS_FIELDS):
            self.losses[index] += losses[name].detach().double() * batch_size
        self.samples += batch_size
        self.unique_conditions += batch["dose"].shape[0]
        self.unique_smiles += batch["node_features"].shape[0]
        condition_to_smiles = auxiliary["condition_to_smiles"].detach()
        smiles_counts = torch.bincount(condition_to_smiles, minlength=batch["node_features"].shape[0])
        self.paired_conditions += smiles_counts[condition_to_smiles].gt(1).sum()
        if self.track_embedding_rank:
            embedding = auxiliary["unique_compound_embedding"].detach().float()
            for local_smiles, smiles_id in enumerate(batch["unique_smiles_ids"].tolist()):
                condition_index = int(
                    torch.nonzero(condition_to_smiles == local_smiles, as_tuple=False)[0, 0].item()
                )
                self.embedding_by_id.setdefault(
                    int(smiles_id),
                    (embedding[condition_index].cpu().numpy().astype(np.float64), 1),
                )
        true_delta = batch["x_treated"] - batch["x_control"]
        self.regression["gene"].update(batch["x_treated"], outputs["pred_gex"])
        self.regression["delta_gene"].update(true_delta, outputs["pred_delta_gene"])
        self.regression["reactome"].update(batch["reactome_target"], outputs["pred_reactome"])
        self.regression["hallmark"].update(batch["hallmark_target"], outputs["pred_hallmark"])

    def compute(self, seconds: float) -> dict[str, float]:
        loss_values = reduce_sum(self.losses.clone())
        samples = reduce_sum(self.samples.clone()).clamp_min(1.0)
        unique = reduce_sum(self.unique_conditions.clone())
        unique_smiles = reduce_sum(self.unique_smiles.clone())
        paired_conditions = reduce_sum(self.paired_conditions.clone())
        result = {name: float((loss_values[index] / samples).item()) for index, name in enumerate(LOSS_FIELDS)}
        for name, accumulator in self.regression.items():
            result.update({f"{name}_{key}": value for key, value in accumulator.compute().items()})
        result["condition_reuse"] = float((samples / unique.clamp_min(1.0)).item())
        result["smiles_reuse"] = float((samples / unique_smiles.clamp_min(1.0)).item())
        result["identity_pair_fraction"] = float((paired_conditions / unique.clamp_min(1.0)).item())
        if self.track_embedding_rank:
            result.update(self._embedding_rank())
        result["seconds"] = float(seconds)
        result["samples_per_second"] = float(samples.item() / max(seconds, 1e-6))
        return result

    def _embedding_rank(self) -> dict[str, float]:
        gathered: list[dict[int, tuple[np.ndarray, int]]] = [self.embedding_by_id]
        if dist.is_initialized():
            gathered = [dict() for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, self.embedding_by_id)
        merged: dict[int, tuple[np.ndarray, int]] = {}
        for values in gathered:
            for smiles_id, value in values.items():
                merged.setdefault(smiles_id, value)
        if len(merged) < 2:
            return {
                "embedding_effective_rank": 1.0,
                "embedding_participation_rank": 1.0,
                "embedding_top10_fraction": 1.0,
            }
        values = np.stack([value for value, _ in merged.values()]).astype(np.float64)
        values -= values.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(values, compute_uv=False)
        energy = np.square(singular)
        probability = energy / max(float(energy.sum()), 1e-12)
        return {
            "embedding_effective_rank": float(
                np.exp(-(probability * np.log(probability + 1e-12)).sum())
            ),
            "embedding_participation_rank": float(
                energy.sum() ** 2 / max(float(np.square(energy).sum()), 1e-12)
            ),
            "embedding_top10_fraction": float(probability[:10].sum()),
        }


def pcc_loss(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    target = target.float().flatten()
    prediction = prediction.float().flatten()
    target = target - target.mean()
    prediction = prediction - prediction.mean()
    denominator = torch.sqrt(target.square().sum() * prediction.square().sum()).clamp_min(1e-8)
    return 1.0 - (target * prediction).sum() / denominator


def gather_embeddings_with_local_grad(embedding: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return embedding
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_count = torch.tensor([embedding.shape[0]], device=embedding.device, dtype=torch.long)
    counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(counts, local_count)
    count_values = [int(value.item()) for value in counts]
    max_count = max(count_values)
    padded = F.pad(embedding, (0, 0, 0, max_count - embedding.shape[0]))
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded.detach())
    gathered[rank] = padded
    return torch.cat([value[:count] for value, count in zip(gathered, count_values)], dim=0)


def feature_regularization(
    values: torch.Tensor, std_floor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    global_values = gather_embeddings_with_local_grad(values.float())
    if global_values.shape[0] <= 1:
        zero = values.new_zeros(())
        return zero, zero
    centered = global_values - global_values.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = F.relu(float(std_floor) - standard_deviation).mean()
    standardized = centered / standard_deviation.detach().clamp_min(0.05)
    covariance = standardized.T @ standardized / max(global_values.shape[0] - 1, 1)
    covariance = covariance - torch.diag_embed(covariance.diagonal())
    strongest_count = min(32, max(covariance.shape[0] - 1, 1))
    strongest = covariance.abs().topk(strongest_count, dim=1).values
    covariance_loss = strongest.square().mean()
    return variance_loss, covariance_loss


def embedding_regularization(
    embedding: torch.Tensor,
    molecule_embedding: torch.Tensor,
    condition_to_smiles: torch.Tensor,
    std_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    condition_to_smiles = condition_to_smiles.long()
    smiles_count = int(condition_to_smiles.max().item()) + 1 if len(condition_to_smiles) else 0
    counts = torch.bincount(condition_to_smiles, minlength=smiles_count).clamp_min(1)
    grouped = embedding.new_zeros((smiles_count, embedding.shape[1]))
    grouped.index_add_(0, condition_to_smiles, embedding)
    grouped = grouped / counts.unsqueeze(1)

    paired = counts.gt(1)
    if paired.any():
        normalized_embedding = F.normalize(embedding, dim=-1)
        normalized_group = F.normalize(grouped, dim=-1)
        pair_mask = paired.index_select(0, condition_to_smiles)
        identity_loss = (
            1.0
            - (normalized_embedding * normalized_group.index_select(0, condition_to_smiles)).sum(dim=-1)
        )[pair_mask].mean()
    else:
        identity_loss = embedding.new_zeros(())

    variance_loss, covariance_loss = feature_regularization(grouped, std_floor)
    molecular_variance_loss, molecular_covariance_loss = feature_regularization(
        F.layer_norm(molecule_embedding, (molecule_embedding.shape[-1],)), std_floor
    )
    return (
        variance_loss,
        covariance_loss,
        molecular_variance_loss,
        molecular_covariance_loss,
        identity_loss,
    )


def compute_loss(
    model: MIPER,
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor | None],
    auxiliary: dict[str, torch.Tensor],
    config: dict[str, float],
) -> dict[str, torch.Tensor]:
    true_delta = batch["x_treated"] - batch["x_control"]
    response_weight = 1.0 + torch.clamp(
        true_delta.abs() / model.gene_delta_std.unsqueeze(0).clamp_min(1e-6),
        max=float(config["response_clip"]),
    )
    response_weight = response_weight / response_weight.mean().detach().clamp_min(1e-6)
    responsive = F.smooth_l1_loss(
        outputs["pred_delta_gene"], true_delta, reduction="none", beta=float(config["huber_beta"])
    )
    embedding = auxiliary["unique_compound_embedding"].float()
    (
        embedding_variance_loss,
        embedding_covariance_loss,
        molecular_variance_loss,
        molecular_covariance_loss,
        identity_consistency_loss,
    ) = (
        embedding_regularization(
            embedding,
            auxiliary["unique_smiles_embedding"],
            auxiliary["condition_to_smiles"],
            float(config["embedding_std_floor"]),
        )
    )
    losses = {
        "gene_loss": F.mse_loss(outputs["pred_gex"], batch["x_treated"]),
        "responsive_delta_loss": (responsive * response_weight).mean(),
        "delta_pcc_loss": pcc_loss(true_delta, outputs["pred_delta_gene"]),
        "reactome_loss": F.mse_loss(outputs["pred_reactome"], batch["reactome_target"]),
        "hallmark_loss": F.mse_loss(outputs["pred_hallmark"], batch["hallmark_target"]),
        "embedding_variance_loss": embedding_variance_loss,
        "embedding_covariance_loss": embedding_covariance_loss,
        "molecular_variance_loss": molecular_variance_loss,
        "molecular_covariance_loss": molecular_covariance_loss,
        "identity_consistency_loss": identity_consistency_loss,
    }
    losses["loss"] = (
        float(config["gene"]) * losses["gene_loss"]
        + float(config["responsive_delta"]) * losses["responsive_delta_loss"]
        + float(config["delta_pcc"]) * losses["delta_pcc_loss"]
        + float(config["reactome"]) * losses["reactome_loss"]
        + float(config["hallmark"]) * losses["hallmark_loss"]
        + float(config["embedding_variance"]) * losses["embedding_variance_loss"]
        + float(config["embedding_covariance"]) * losses["embedding_covariance_loss"]
        + float(config["molecular_variance"]) * losses["molecular_variance_loss"]
        + float(config["molecular_covariance"]) * losses["molecular_covariance_loss"]
        + float(config["identity_consistency"]) * losses["identity_consistency_loss"]
    )
    return losses


class GroupedTrainingModel(nn.Module):
    def __init__(self, model: MIPER):
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, torch.Tensor]):
        return self.model.forward_grouped(
            batch["x_control"],
            batch["node_features"],
            batch["mask"],
            batch["adjacency"],
            batch["distance"],
            batch["dose"],
            batch["time"],
            batch["condition_inverse"],
            batch["condition_to_smiles"],
        )


def build_model(config: dict[str, Any], device: torch.device) -> MIPER:
    data = config["data"]
    stats = np.load(Path(data["cache_dir"]) / "control_stats.npz")
    missing = {"treated_mean", "treated_std"}.difference(stats.files)
    if missing:
        raise RuntimeError(
            f"MIPER statistics are missing {sorted(missing)}; run the prepare command first"
        )
    model_config = dict(config["model"])
    model_config.pop("gene_dim", None)
    model = MIPER.from_files(
        genes_file=Path(data["lincs_dir"]) / "genes.txt",
        pathway_dir=data["pathway_dir"],
        treated_mean=torch.from_numpy(stats["treated_mean"]),
        treated_std=torch.from_numpy(stats["treated_std"]),
        **model_config,
    )
    return model.to(device)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def run_epoch(
    module: GroupedTrainingModel | DDP,
    model: MIPER,
    loader: DataLoader,
    device: torch.device,
    loss_config: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    rank: int,
    accumulate_steps: int,
    grad_clip: float,
    heartbeat_steps: int,
    max_steps: int | None = None,
    track_embedding_rank: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    module.train(training)
    stats = EpochStats(device, track_embedding_rank=track_embedding_rank)
    steps = min(len(loader), max_steps) if max_steps else len(loader)
    start_time = time.perf_counter()
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        if max_steps and step > max_steps:
            break
        batch = move_batch(batch, device)
        synchronize = not training or step % accumulate_steps == 0 or step == steps
        sync_context = nullcontext() if synchronize or not isinstance(module, DDP) else module.no_sync()
        with sync_context:
            with torch.set_grad_enabled(training), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=False,
            ):
                outputs, auxiliary = module(batch)
                losses = compute_loss(
                    model,
                    batch,
                    outputs,
                    auxiliary,
                    loss_config,
                )
                backward_loss = losses["loss"] / accumulate_steps
            if training:
                backward_loss.backward()
        if training and synchronize:
            torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        stats.update(batch, outputs, auxiliary, losses)
        if rank == 0 and heartbeat_steps and (step % heartbeat_steps == 0 or step == steps):
            elapsed = time.perf_counter() - start_time
            samples = step * int(batch["x_control"].shape[0]) * int(os.environ.get("WORLD_SIZE", "1"))
            print(
                f"heartbeat epoch={epoch} step={step}/{steps} loss={float(losses['loss'].detach()):.6f} "
                f"samples_per_second={samples / max(elapsed, 1e-6):.1f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    result = stats.compute(elapsed)
    attention_mix = torch.stack(
        [block.attention_mix().detach().float().mean(dim=0) for block in model.molecular_encoder.blocks]
    ).mean(dim=0)
    result["attention_content"] = float(attention_mix[0].item())
    result["attention_bond"] = float(attention_mix[1].item())
    result["attention_distance"] = float(attention_mix[2].item())
    return result


def score_metrics(
    metrics: dict[str, float],
    selection: dict[str, Any],
    baseline: dict[str, float] | None = None,
) -> float:
    baseline = baseline or selection["baseline"]
    weight = selection["score_weights"]
    scale = selection["score_scales"]
    return float(
        weight["gene_pcc"] * (metrics["gene_pcc"] - baseline["gene_pcc"]) / scale["gene_pcc"]
        + weight["delta_gene_pcc"]
        * (metrics["delta_gene_pcc"] - baseline["delta_gene_pcc"])
        / scale["delta_gene_pcc"]
        + weight["reactome_pcc"] * (metrics["reactome_pcc"] - baseline["reactome_pcc"]) / scale["reactome_pcc"]
        + weight["hallmark_pcc"] * (metrics["hallmark_pcc"] - baseline["hallmark_pcc"]) / scale["hallmark_pcc"]
        + weight["gene_rmse"] * (baseline["gene_rmse"] - metrics["gene_rmse"]) / scale["gene_rmse"]
    )


def load_transfer_baseline(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["selection"]["transfer_baseline_file"])
    if not path.exists():
        raise FileNotFoundError(f"V31 transfer baseline is missing: {path}")
    baseline = json.loads(path.read_text())
    required = {"random_valid", "drug_ood_valid", "scaffold_ood_valid"}
    if set(baseline.get("splits", {})) != required:
        raise RuntimeError(f"Unexpected V31 transfer baseline splits: {baseline.get('splits', {}).keys()}")
    return baseline


def eligible_checkpoint(
    random_metrics: dict[str, float],
    drug_metrics: dict[str, float],
    scaffold_metrics: dict[str, float],
    baseline: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    v31 = baseline["splits"]["random_valid"]
    tolerance = selection["random_guard_tolerance"]
    checks = {
        "gene_rmse": random_metrics["gene_rmse"] <= v31["gene_rmse"] + float(tolerance["gene_rmse"]),
        "gene_pcc": random_metrics["gene_pcc"] >= v31["gene_pcc"] - float(tolerance["gene_pcc"]),
        "delta_gene_pcc": random_metrics["delta_gene_pcc"]
        >= v31["delta_gene_pcc"] - float(tolerance["delta_gene_pcc"]),
        "reactome_pcc": random_metrics["reactome_pcc"]
        >= v31["reactome_pcc"] - float(tolerance["reactome_pcc"]),
        "hallmark_pcc": random_metrics["hallmark_pcc"]
        >= v31["hallmark_pcc"] - float(tolerance["hallmark_pcc"]),
    }
    rank_baseline = baseline["ood_embedding_rank"]
    rank_fraction = float(selection["embedding_rank_min_fraction"])
    top10_margin = float(selection["embedding_top10_max_margin"])
    for name, metrics in (("drug", drug_metrics), ("scaffold", scaffold_metrics)):
        checks[f"{name}_rank"] = metrics["embedding_effective_rank"] >= (
            rank_fraction * rank_baseline["effective_rank_entropy"]
        )
        checks[f"{name}_top10"] = metrics["embedding_top10_fraction"] <= (
            rank_baseline["top10_fraction"] + top10_margin
        )
    return all(checks.values()), checks


def append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, extrasaction="ignore", delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in ROW_FIELDS})


def metric_row(epoch: int, split: str, metrics: dict[str, float], score: float, lr: float) -> dict[str, Any]:
    return {"epoch": epoch, "split": split, **metrics, "score": score, "lr": lr}


def selection_improved(
    best: dict[str, Any], score: float, eligible: bool, min_delta: float
) -> bool:
    return bool(
        (eligible and not bool(best.get("eligible", False)))
        or (
            eligible == bool(best.get("eligible", False))
            and score > float(best["score"]) + min_delta
        )
    )


def save_checkpoint(
    path: Path,
    model: MIPER,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    best: dict[str, Any],
    no_improvement_epochs: int,
    include_optimizer: bool = True,
) -> None:
    temporary = path.with_suffix(".tmp")
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best": best,
        "no_improvement_epochs": no_improvement_epochs,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: MIPER,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> dict[str, Any]:
    checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_data["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint_data["scheduler"])
    return checkpoint_data


def divergent(metrics: dict[str, float], config: dict[str, Any], epoch: int) -> bool:
    limits = config["selection"]["divergence"]
    if epoch < int(limits["after_epoch"]):
        return False
    return bool(
        metrics["gene_pcc"] < limits["gene_pcc_min"]
        or metrics["delta_gene_pcc"] < limits["delta_gene_pcc_min"]
        or metrics["reactome_pcc"] < limits["reactome_pcc_min"]
        or metrics["hallmark_pcc"] < limits["hallmark_pcc_min"]
        or not all(math.isfinite(metrics[name]) for name in METRIC_FIELDS)
    )


def build_training_objects(config: dict[str, Any], device: torch.device, world_size: int):
    model = build_model(config, device)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    wrapper = GroupedTrainingModel(model).to(device)
    module: GroupedTrainingModel | DDP = wrapper
    if world_size > 1:
        module = DDP(wrapper, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
        fused=False,
        foreach=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["train"]["scheduler_factor"]),
        patience=int(config["train"]["scheduler_patience"]),
        min_lr=float(config["train"]["min_lr"]),
    )
    return model, module, optimizer, scheduler


def run_smoke(config: dict[str, Any]) -> None:
    rank, _, world_size, device = setup_distributed(int(config["train"]["seed"]))
    distributed_barrier()
    model, module, optimizer, scheduler = build_training_objects(config, device, world_size)
    train_steps = int(config["smoke"]["train_steps"])
    valid_steps = int(config["smoke"]["valid_steps"])
    train_loader, sampler = make_loader(
        config,
        "train",
        rank,
        world_size,
        True,
        limit=None,
    )
    valid_loader, _ = make_loader(
        config,
        "random_valid",
        rank,
        world_size,
        False,
        limit=valid_steps * int(config["train"]["eval_batch_size_per_gpu"]) * world_size,
    )
    assert sampler is not None
    sampler.set_epoch(0)
    train_metrics = run_epoch(
        module,
        model,
        train_loader,
        device,
        config["loss"],
        optimizer,
        0,
        rank,
        int(config["train"]["accumulate_steps"]),
        float(config["train"]["grad_clip"]),
        int(config["train"]["heartbeat_steps"]),
        train_steps,
    )
    valid_metrics = run_epoch(
        module,
        model,
        valid_loader,
        device,
        config["loss"],
        None,
        0,
        rank,
        1,
        float(config["train"]["grad_clip"]),
        0,
        valid_steps,
        track_embedding_rank=True,
    )
    smoke_dir = Path(config["train"]["run_dir"]) / "smoke"
    if rank == 0:
        smoke_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = smoke_dir / "checkpoint.pt"
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, 0, {"epoch": 0, "score": 0.0}, 7)
        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" not in reloaded or int(reloaded.get("no_improvement_epochs", -1)) != 7:
            raise RuntimeError("Smoke checkpoint reload failed")
        attention_bond = min(
            float(block.attention_mix()[:, 1].min().item()) for block in model.molecular_encoder.blocks
        )
        forbidden_fragments = ("prior_basis", "discovery_basis", "known_correction")
        forbidden_state = [
            name for name in model.state_dict() if any(value in name for value in forbidden_fragments)
        ]
        checks = {
            "finite_train": all(math.isfinite(value) for value in train_metrics.values()),
            "finite_valid": all(math.isfinite(value) for value in valid_metrics.values()),
            "bond_floor": attention_bond >= 0.2 - 1e-6,
            "forbidden_state_absent": not forbidden_state,
            "identity_pairs": train_metrics["identity_pair_fraction"] > 0.0,
            "checkpoint_reload": True,
            "patience_counter_reload": True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"MIPER smoke checks failed: {checks}")
        write_json(
            smoke_dir / "result.json",
            {
                "status": "passed",
                "parameters": parameter_summary(model),
                "train": train_metrics,
                "valid": valid_metrics,
                "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
                "checks": checks,
            },
        )
        print(json.dumps(json.loads((smoke_dir / "result.json").read_text()), indent=2), flush=True)
    finish_distributed()


def run_train(config: dict[str, Any]) -> None:
    rank, _, world_size, device = setup_distributed(int(config["train"]["seed"]))
    run_dir = Path(config["train"]["run_dir"])
    result_dir = Path(config["train"]["result_dir"])
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config_used.yaml").open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    distributed_barrier()
    model, module, optimizer, scheduler = build_training_objects(config, device, world_size)
    train_loader, sampler = make_loader(config, "train", rank, world_size, True)
    validation_loaders = {
        split: make_loader(config, split, rank, world_size, False)[0]
        for split in ("random_valid", "drug_ood_valid", "scaffold_ood_valid")
    }
    transfer_baseline = load_transfer_baseline(config)
    assert sampler is not None

    metrics_path = run_dir / "metrics.tsv"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    status_path = run_dir / "status.json"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best = {
        "epoch": 0,
        "score": -float("inf"),
        "metrics": {},
        "eligible": False,
        "checks": {},
    }
    no_improvement_epochs = 0
    if last_path.exists():
        checkpoint_data = load_checkpoint(last_path, model, optimizer, scheduler)
        start_epoch = int(checkpoint_data["epoch"]) + 1
        best = dict(checkpoint_data["best"])
        no_improvement_epochs = int(
            checkpoint_data.get("no_improvement_epochs", max(0, start_epoch - 1 - int(best["epoch"])))
        )
        if metrics_path.exists():
            history = pd.read_csv(metrics_path, sep="\t").to_dict("records")
        if rank == 0:
            print(
                f"resume epoch={start_epoch} best={best['epoch']} "
                f"no_improvement={no_improvement_epochs}",
                flush=True,
            )

    divergence_count = 0
    trained_through = start_epoch - 1
    stop_reason = "running"
    base_lr = float(config["train"]["learning_rate"])
    improvement_patience = int(config["train"]["improvement_patience"])
    improvement_min_delta = float(config["train"]["improvement_min_delta"])
    if improvement_patience <= 0 or improvement_min_delta < 0:
        raise ValueError("improvement_patience must be positive and improvement_min_delta nonnegative")
    epoch = start_epoch
    while True:
        trained_through = epoch
        warmup_epochs = int(config["train"]["warmup_epochs"])
        if epoch <= warmup_epochs:
            warmup_lr = base_lr * epoch / warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            module,
            model,
            train_loader,
            device,
            config["loss"],
            optimizer,
            epoch,
            rank,
            int(config["train"]["accumulate_steps"]),
            float(config["train"]["grad_clip"]),
            int(config["train"]["heartbeat_steps"]),
        )
        validation_metrics = {
            split: run_epoch(
                module,
                model,
                loader,
                device,
                config["loss"],
                None,
                epoch,
                rank,
                1,
                float(config["train"]["grad_clip"]),
                0,
                track_embedding_rank=split != "random_valid",
            )
            for split, loader in validation_loaders.items()
        }
        split_scores = {
            split: score_metrics(
                metrics,
                config["selection"],
                transfer_baseline["splits"][split],
            )
            for split, metrics in validation_metrics.items()
        }
        score = 0.5 * (
            split_scores["drug_ood_valid"] + split_scores["scaffold_ood_valid"]
        )
        eligible, eligibility_checks = eligible_checkpoint(
            validation_metrics["random_valid"],
            validation_metrics["drug_ood_valid"],
            validation_metrics["scaffold_ood_valid"],
            transfer_baseline,
            config["selection"],
        )
        if epoch > warmup_epochs:
            scheduler.step(score)
        lr = float(optimizer.param_groups[0]["lr"])
        train_row = metric_row(epoch, "train", train_metrics, score_metrics(train_metrics, config["selection"]), lr)
        validation_rows = [
            metric_row(epoch, split, metrics, split_scores[split], lr)
            for split, metrics in validation_metrics.items()
        ]
        history.extend([train_row, *validation_rows])
        replace = selection_improved(best, score, eligible, improvement_min_delta)
        if replace:
            best = {
                "epoch": epoch,
                "score": score,
                "metrics": validation_metrics,
                "eligible": eligible,
                "checks": eligibility_checks,
            }
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        all_finite = all(
            all(math.isfinite(metrics[name]) for name in METRIC_FIELDS)
            for metrics in validation_metrics.values()
        )
        if not all_finite:
            raise RuntimeError(f"non-finite validation metric at epoch {epoch}")
        divergence_count = (
            divergence_count + 1
            if divergent(validation_metrics["random_valid"], config, epoch) or not all_finite
            else 0
        )

        if rank == 0:
            append_row(metrics_path, train_row)
            for row in validation_rows:
                append_row(metrics_path, row)
            if replace:
                save_checkpoint(
                    best_path, model, optimizer, scheduler, epoch, best, no_improvement_epochs,
                    include_optimizer=False,
                )
            save_checkpoint(last_path, model, optimizer, scheduler, epoch, best, no_improvement_epochs)
            epoch_seconds = train_metrics["seconds"] + sum(
                metrics["seconds"] for metrics in validation_metrics.values()
            )
            epochs_until_stop = max(improvement_patience - no_improvement_epochs, 0)
            minimum_eta_hours = epoch_seconds * epochs_until_stop / 3600.0
            write_json(
                status_path,
                {
                    "status": "running",
                    "epoch": epoch,
                    "best_epoch": best["epoch"],
                    "best_score": best["score"],
                    "no_improvement_epochs": no_improvement_epochs,
                    "improvement_patience": improvement_patience,
                    "improvement_min_delta": improvement_min_delta,
                    "lr": lr,
                    "epoch_seconds": epoch_seconds,
                    "minimum_estimated_hours_to_stop": minimum_eta_hours,
                    "eligible": eligible,
                    "eligibility_checks": eligibility_checks,
                    "validation": validation_metrics,
                },
            )
            print(
                f"epoch={epoch} ood_score={score:.5f} eligible={eligible} "
                f"random_pcc={validation_metrics['random_valid']['gene_pcc']:.5f} "
                f"drug_delta_pcc={validation_metrics['drug_ood_valid']['delta_gene_pcc']:.5f} "
                f"scaffold_delta_pcc={validation_metrics['scaffold_ood_valid']['delta_gene_pcc']:.5f} "
                f"scaffold_hallmark_pcc={validation_metrics['scaffold_ood_valid']['hallmark_pcc']:.5f} "
                f"epoch_hours={epoch_seconds / 3600:.2f} "
                f"no_improvement={no_improvement_epochs}/{improvement_patience} "
                f"minimum_eta_hours={minimum_eta_hours:.1f} best={best['epoch']}",
                flush=True,
            )
        if dist.is_initialized():
            payload = [best, divergence_count, no_improvement_epochs]
            dist.broadcast_object_list(payload, src=0)
            best, divergence_count, no_improvement_epochs = payload
        if bool(config["train"].get("metric_deficit_early_stop", True)) and divergence_count >= int(
            config["train"]["divergence_patience"]
        ):
            stop_reason = "persistent_metric_deficit"
            break
        if no_improvement_epochs >= improvement_patience:
            stop_reason = "selection_no_improvement_patience"
            break
        epoch += 1

    distributed_barrier()
    finalize(
        config, model, module, device, rank, world_size, best_path, metrics_path,
        trained_through, stop_reason, no_improvement_epochs,
    )
    finish_distributed()


def finalize(
    config: dict[str, Any],
    model: MIPER,
    module: GroupedTrainingModel | DDP,
    device: torch.device,
    rank: int,
    world_size: int,
    best_path: Path,
    metrics_path: Path,
    trained_through: int,
    stop_reason: str,
    no_improvement_epochs: int,
) -> None:
    checkpoint_data = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_data["model"])
    best_epoch = int(checkpoint_data["best"]["epoch"])
    best_score = float(checkpoint_data["best"]["score"])
    validation_metrics = {}
    for split in ("random_valid", "drug_ood_valid", "scaffold_ood_valid"):
        loader, _ = make_loader(config, split, rank, world_size, False)
        validation_metrics[split] = run_epoch(
            module,
            model,
            loader,
            device,
            config["loss"],
            None,
            best_epoch,
            rank,
            1,
            1.0,
            0,
            track_embedding_rank=split != "random_valid",
        )
    test_loader, _ = make_loader(config, "test", rank, world_size, False)
    test_metrics = run_epoch(module, model, test_loader, device, config["loss"], None, best_epoch, rank, 1, 1.0, 0)
    if rank != 0:
        return

    run_dir = Path(config["train"]["run_dir"])
    result_dir = Path(config["train"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(metrics_path, sep="\t")
    full_metrics_sha256 = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    ood_trajectory = (
        history[history["split"].isin({"drug_ood_valid", "scaffold_ood_valid"})]
        .groupby("epoch", as_index=False)["score"]
        .mean()
        .rename(columns={"score": "ood_score"})
        .to_dict("records")
    )
    best_train = history[(history["epoch"] == best_epoch) & (history["split"] == "train")]
    rows = []
    if not best_train.empty:
        rows.append({"split": "train", **best_train.iloc[-1].to_dict()})
    rows.extend(
        [{"split": split, "epoch": best_epoch, **metrics} for split, metrics in validation_metrics.items()]
    )
    rows.append({"split": "test", "epoch": best_epoch, **test_metrics})
    pd.DataFrame(rows).to_csv(result_dir / "final_metrics.tsv", sep="\t", index=False)
    compact = {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "epoch": best_epoch,
        "score": best_score,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "selection_eligible": bool(checkpoint_data["best"].get("eligible", False)),
        "eligibility_checks": checkpoint_data["best"].get("checks", {}),
        "architecture": "MIPER-v1",
        "model_config": dict(config["model"]),
        "parameters": parameter_summary(model),
    }
    compact_path = best_path.with_suffix(".compact.pt")
    torch.save(compact, compact_path)
    os.replace(compact_path, best_path)
    history[history["epoch"] <= best_epoch].to_csv(metrics_path, sep="\t", index=False)
    (run_dir / "last.pt").unlink(missing_ok=True)
    shutil.rmtree(run_dir / "smoke", ignore_errors=True)

    transfer_baseline = load_transfer_baseline(config)
    comparisons = {
        split: {
            name: metrics[name] - float(transfer_baseline["splits"][split][name])
            for name in config["selection"]["baseline"]
        }
        for split, metrics in validation_metrics.items()
    }
    summary = {
        "status": "completed" if checkpoint_data["best"].get("eligible", False) else "completed_ineligible",
        "trained_through_epoch": trained_through,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "stop_reason": stop_reason,
        "no_improvement_epochs": no_improvement_epochs,
        "improvement_patience": int(config["train"]["improvement_patience"]),
        "improvement_min_delta": float(config["train"]["improvement_min_delta"]),
        "selection_eligible": bool(checkpoint_data["best"].get("eligible", False)),
        "eligibility_checks": checkpoint_data["best"].get("checks", {}),
        "architecture": "MIPER-v1",
        "model_config": dict(config["model"]),
        "parameters": parameter_summary(model),
        "validation": validation_metrics,
        "test": test_metrics,
        "delta_from_v31": comparisons,
        "ood_trajectory": ood_trajectory,
    }
    write_json(result_dir / "summary.json", summary)
    write_json(run_dir / "status.json", summary)
    write_json(
        run_dir / "training_evidence.json",
        {
            "trained_through_epoch": trained_through,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "stop_reason": stop_reason,
            "no_improvement_epochs": no_improvement_epochs,
            "improvement_patience": int(config["train"]["improvement_patience"]),
            "improvement_min_delta": float(config["train"]["improvement_min_delta"]),
            "full_metrics_rows": int(len(history)),
            "full_metrics_sha256_before_trim": full_metrics_sha256,
            "ood_trajectory": ood_trajectory,
        },
    )
    report = [
        "# MIPER LINCS pretraining report",
        "",
        f"- Seed: `{config['train']['seed']}`",
        f"- Trained through: `{trained_through}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Stop reason: `{stop_reason}`",
        "",
        f"- Eligible for downstream: `{summary['selection_eligible']}`",
        "",
    ]
    for split, metrics in validation_metrics.items():
        report.extend(
            [
                f"## {split}",
                "",
                "| metric | V31 | MIPER | delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for name in config["selection"]["baseline"]:
            value = float(transfer_baseline["splits"][split][name])
            report.append(
                f"| {name} | {value:.6f} | {metrics[name]:.6f} | {metrics[name] - value:+.6f} |"
            )
    report.extend(
        [
            "",
            "## Test",
            "",
            f"- Gene RMSE/PCC: `{test_metrics['gene_rmse']:.6f}` / `{test_metrics['gene_pcc']:.6f}`",
            f"- DeltaGene PCC: `{test_metrics['delta_gene_pcc']:.6f}`",
            f"- Reactome PCC: `{test_metrics['reactome_pcc']:.6f}`",
            f"- Hallmark PCC: `{test_metrics['hallmark_pcc']:.6f}`",
            "",
            "The unchanged V31 GDSC2 fold0/fold1 validation starts only when selection_eligible is true.",
        ]
    )
    (result_dir / "experiment_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MIPER LINCS pretraining")
    parser.add_argument("--config", default=str(RELEASE_ROOT / "configs" / "pretrain.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("smoke")
    subparsers.add_parser("train")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        print(json.dumps(prepare_data_index(config), indent=2))
    elif args.command == "smoke":
        run_smoke(config)
    elif args.command == "train":
        run_train(config)


if __name__ == "__main__":
    main()
