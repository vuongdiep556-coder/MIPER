from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianRBF(nn.Module):
    def __init__(self, channels: int = 64, cutoff: float = 12.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, channels)
        spacing = float(centers[1] - centers[0]) if channels > 1 else cutoff
        self.register_buffer("centers", centers)
        self.register_buffer("inverse_width", torch.tensor(1.0 / max(spacing * spacing, 1e-6)))
        self.cutoff = float(cutoff)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        distance = distance.clamp(min=0.0, max=self.cutoff).unsqueeze(-1)
        return torch.exp(-self.inverse_width * (distance - self.centers).square())


class ConditionFeatureEncoder(nn.Module):
    """Encode dose and time without changing the public input fields."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, dose: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        dose = dose.reshape(dose.shape[0], -1)[:, :1].clamp_min(0.0)
        time = time.reshape(time.shape[0], -1)[:, :1].clamp_min(0.0)
        features = torch.cat(
            [
                dose,
                torch.log1p(dose),
                torch.sqrt(dose + 1e-6),
                time,
                torch.log1p(time),
                dose * time,
            ],
            dim=-1,
        )
        return self.network(features)


class EdgeGatedGraphBlock(nn.Module):
    """Topology-preserving attention over content, bonds, and distance."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        heads: int,
        rbf_dim: int,
        dropout: float,
        bond_floor: float = 0.2,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.distance_gate = nn.Linear(rbf_dim, heads, bias=False)
        self.mix_logits = nn.Parameter(torch.zeros(heads, 3))
        self.bond_floor = float(bond_floor)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message_out = nn.Linear(hidden_dim, hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_in = nn.Linear(hidden_dim, ffn_dim * 2)
        self.ffn_out = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.message_scale = nn.Parameter(torch.full((hidden_dim,), 0.1))
        self.ffn_scale = nn.Parameter(torch.full((hidden_dim,), 0.1))

    def forward(
        self,
        node_state: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        adjacency: torch.Tensor,
        distance_rbf: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.message_norm(node_state)
        query = self.query(normalized).reshape(
            normalized.shape[0], normalized.shape[1], self.heads, self.head_dim
        )
        key = self.key(normalized).reshape(
            normalized.shape[0], normalized.shape[1], self.heads, self.head_dim
        )
        content_logits = torch.einsum("bihd,bjhd->bijh", query, key) / math.sqrt(self.head_dim)
        content_logits = content_logits.masked_fill(
            ~edge_mask.unsqueeze(-1), torch.finfo(content_logits.dtype).min
        )
        content_probability = torch.softmax(content_logits, dim=2)

        valid_identity = torch.diag_embed(node_mask.to(adjacency.dtype))
        bond_weight = adjacency.abs() + valid_identity
        bond_weight = bond_weight * edge_mask.to(bond_weight.dtype)
        bond_probability = bond_weight / bond_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        bond_probability = bond_probability.unsqueeze(-1)

        distance_logits = self.distance_gate(distance_rbf)
        distance_logits = distance_logits.masked_fill(
            ~edge_mask.unsqueeze(-1), torch.finfo(distance_logits.dtype).min
        )
        distance_probability = torch.softmax(distance_logits, dim=2)

        raw_mix = torch.softmax(self.mix_logits, dim=-1)
        bond_mix = self.bond_floor + (1.0 - self.bond_floor) * raw_mix[:, 1]
        residual_mix = 1.0 - bond_mix
        nonbond = raw_mix[:, [0, 2]]
        nonbond = nonbond / nonbond.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        content_mix = residual_mix * nonbond[:, 0]
        distance_mix = residual_mix * nonbond[:, 1]
        gate = (
            content_probability * content_mix.view(1, 1, 1, -1)
            + bond_probability * bond_mix.view(1, 1, 1, -1)
            + distance_probability * distance_mix.view(1, 1, 1, -1)
        )
        values = self.value(normalized).reshape(
            normalized.shape[0], normalized.shape[1], self.heads, self.head_dim
        )
        message = torch.einsum("bijh,bjhd->bihd", gate, values).flatten(-2)
        message = self.message_out(message)

        node_state = node_state + self.message_scale * self.dropout(F.silu(message))

        ffn_input = self.ffn_in(self.ffn_norm(node_state))
        ffn_value, ffn_gate = ffn_input.chunk(2, dim=-1)
        ffn_output = self.ffn_out(ffn_value * F.silu(ffn_gate))
        node_state = node_state + self.ffn_scale * self.dropout(ffn_output)
        return node_state * node_mask.unsqueeze(-1)

    def attention_mix(self) -> torch.Tensor:
        raw_mix = torch.softmax(self.mix_logits, dim=-1)
        bond = self.bond_floor + (1.0 - self.bond_floor) * raw_mix[:, 1]
        residual = 1.0 - bond
        nonbond = raw_mix[:, [0, 2]]
        nonbond = nonbond / nonbond.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.stack([residual * nonbond[:, 0], bond, residual * nonbond[:, 1]], dim=-1)


class MolecularConditionEncoder(nn.Module):
    """A clean-room molecular encoder based on edge-gated message passing."""

    def __init__(
        self,
        atom_dim: int = 28,
        hidden_dim: int = 768,
        ffn_dim: int = 3072,
        output_dim: int = 768,
        layers: int = 10,
        heads: int = 8,
        rbf_dim: int = 64,
        dropout: float = 0.1,
        activation_checkpointing: bool = True,
    ):
        super().__init__()
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.condition_encoder = ConditionFeatureEncoder(hidden_dim)
        self.distance_rbf = GaussianRBF(rbf_dim)
        self.blocks = nn.ModuleList(
            [
                EdgeGatedGraphBlock(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    heads=heads,
                    rbf_dim=rbf_dim,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.pool_value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.SiLU(),
            nn.LayerNorm(output_dim),
        )
        self.activation_checkpointing = bool(activation_checkpointing)

    def _valid_atom_mask(
        self, mask: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        # Cached molecular features include an isolated dummy node. Excluding
        # degree-zero nodes keeps it out of both propagation and pooling.
        valid = mask.bool() & adjacency.abs().sum(dim=-1).gt(0)
        empty_graph = ~valid.any(dim=1)
        if empty_graph.any():
            valid = valid.clone()
            valid[empty_graph] = mask.bool()[empty_graph]
        return valid

    def encode_molecule(
        self,
        node_features: torch.Tensor,
        mask: torch.Tensor,
        adjacency: torch.Tensor,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        node_mask = self._valid_atom_mask(mask, adjacency)
        # Global valid-atom mixing keeps unseen compounds geometrically smooth;
        # adjacency remains an explicit learned bias and distance remains gated.
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        node_state = self.atom_projection(node_features) * node_mask.unsqueeze(-1)
        identity_state = node_state
        distance_rbf = self.distance_rbf(distance)

        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                node_state = checkpoint(
                    block,
                    node_state,
                    node_mask,
                    edge_mask,
                    adjacency,
                    distance_rbf,
                    use_reentrant=False,
                )
            else:
                node_state = block(node_state, node_mask, edge_mask, adjacency, distance_rbf)

        atom_weights = node_mask.unsqueeze(-1).to(node_state.dtype)
        atom_count = atom_weights.sum(dim=1).clamp_min(1.0)
        final_mean = (node_state * atom_weights).sum(dim=1) / atom_count
        identity_mean = (identity_state * atom_weights).sum(dim=1) / atom_count
        pooled = (
            F.layer_norm(final_mean, (final_mean.shape[-1],))
            + F.layer_norm(identity_mean, (identity_mean.shape[-1],))
        ) / math.sqrt(2.0)
        return self.pool_value(pooled)

    def combine_condition(
        self, molecule_embedding: torch.Tensor, dose: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        context = self.condition_encoder(dose, time)
        molecule_embedding = F.layer_norm(molecule_embedding, (molecule_embedding.shape[-1],))
        condition = self.output(torch.cat([molecule_embedding, 0.5 * context], dim=-1))
        if condition.shape[-1] == molecule_embedding.shape[-1]:
            condition = condition + molecule_embedding
        return condition

    def forward(
        self,
        node_features: torch.Tensor,
        mask: torch.Tensor,
        adjacency: torch.Tensor,
        distance: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        molecule_embedding = self.encode_molecule(node_features, mask, adjacency, distance)
        return self.combine_condition(molecule_embedding, dose, time)


class MIPER(nn.Module):
    """Molecular Identity-Preserving Expression Response encoder."""

    def __init__(
        self,
        gene_dim: int = 978,
        atom_dim: int = 28,
        molecular_hidden: int = 1024,
        molecular_ffn: int = 4096,
        condition_dim: int = 1024,
        molecular_layers: int = 8,
        molecular_heads: int = 16,
        reactome_dim: int = 472,
        hallmark_dim: int = 49,
        dropout: float = 0.1,
        activation_checkpointing: bool = True,
        treated_mean: torch.Tensor | None = None,
        treated_std: torch.Tensor | None = None,
        gene_delta_mean: torch.Tensor | None = None,
        gene_delta_std: torch.Tensor | None = None,
        reactome_membership: torch.Tensor | None = None,
        hallmark_membership: torch.Tensor | None = None,
        reactome_raw_mean: torch.Tensor | None = None,
        reactome_raw_std: torch.Tensor | None = None,
        hallmark_raw_mean: torch.Tensor | None = None,
        hallmark_raw_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gene_dim = int(gene_dim)
        self.molecular_encoder = MolecularConditionEncoder(
            atom_dim=atom_dim,
            hidden_dim=molecular_hidden,
            ffn_dim=molecular_ffn,
            output_dim=condition_dim,
            layers=molecular_layers,
            heads=molecular_heads,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
        )
        self.compound_head = nn.Sequential(
            nn.Linear(condition_dim, gene_dim, bias=False),
            nn.BatchNorm1d(gene_dim),
            nn.LeakyReLU(negative_slope=0.05),
        )
        self.compound_identity_scale = 0.5
        self.gene_input_norm = nn.LayerNorm(gene_dim)
        self.delta_norm = nn.LayerNorm(gene_dim)
        self.delta_head = nn.Linear(gene_dim, gene_dim)
        self.reactome_dim = int(reactome_dim)
        self.reactome_log_scale = nn.Parameter(torch.zeros(reactome_dim))
        self.reactome_bias = nn.Parameter(torch.zeros(reactome_dim))
        self.hallmark_log_scale = nn.Parameter(torch.zeros(hallmark_dim))
        self.hallmark_bias = nn.Parameter(torch.zeros(hallmark_dim))

        def vector(value: torch.Tensor | None, size: int, default: float) -> torch.Tensor:
            return torch.full((size,), default) if value is None else value.detach().float().reshape(size)

        self.register_buffer("treated_mean", vector(treated_mean, gene_dim, 0.0))
        self.register_buffer("treated_std", vector(treated_std, gene_dim, 1.0).clamp_min(1e-6))
        self.register_buffer("gene_delta_mean", vector(gene_delta_mean, gene_dim, 0.0))
        self.register_buffer("gene_delta_std", vector(gene_delta_std, gene_dim, 1.0).clamp_min(1e-6))
        self.register_buffer(
            "reactome_membership",
            torch.zeros(reactome_dim, gene_dim) if reactome_membership is None else reactome_membership.float(),
        )
        self.register_buffer(
            "hallmark_membership",
            torch.zeros(hallmark_dim, gene_dim) if hallmark_membership is None else hallmark_membership.float(),
        )
        self.register_buffer("reactome_raw_mean", vector(reactome_raw_mean, reactome_dim, 0.0))
        self.register_buffer("reactome_raw_std", vector(reactome_raw_std, reactome_dim, 1.0).clamp_min(1e-6))
        self.register_buffer("hallmark_raw_mean", vector(hallmark_raw_mean, hallmark_dim, 0.0))
        self.register_buffer("hallmark_raw_std", vector(hallmark_raw_std, hallmark_dim, 1.0).clamp_min(1e-6))

    def fixed_reactome(self, delta_gene_z: torch.Tensor) -> torch.Tensor:
        raw = delta_gene_z @ self.reactome_membership.T
        return (raw - self.reactome_raw_mean) / self.reactome_raw_std

    def fixed_hallmark(self, delta_gene_z: torch.Tensor) -> torch.Tensor:
        raw = delta_gene_z @ self.hallmark_membership.T
        return (raw - self.hallmark_raw_mean) / self.hallmark_raw_std

    def project_compound(self, condition: torch.Tensor) -> torch.Tensor:
        learned = self.compound_head(condition)
        if condition.shape[-1] >= self.gene_dim:
            identity = condition[:, : self.gene_dim]
        else:
            identity = F.pad(condition, (0, self.gene_dim - condition.shape[-1]))
        identity = F.layer_norm(identity, (self.gene_dim,))
        return learned + self.compound_identity_scale * identity

    def _forward_impl(
        self,
        x_control: torch.Tensor,
        node_features: torch.Tensor,
        mask: torch.Tensor,
        adjacency: torch.Tensor,
        distance: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor]]:
        molecule_embedding = self.molecular_encoder.encode_molecule(
            node_features, mask, adjacency, distance
        )
        condition = self.molecular_encoder.combine_condition(molecule_embedding, dose, time)
        compound_embedding = self.project_compound(condition)
        outputs, auxiliary = self._forward_from_compound(x_control, compound_embedding)
        auxiliary["condition"] = condition
        auxiliary["unique_condition"] = condition
        auxiliary["unique_compound_embedding"] = compound_embedding
        auxiliary["unique_smiles_embedding"] = molecule_embedding
        auxiliary["condition_to_smiles"] = torch.arange(
            condition.shape[0], device=condition.device, dtype=torch.long
        )
        return outputs, auxiliary

    def _forward_from_compound(
        self,
        x_control: torch.Tensor,
        compound_embedding: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor]]:
        # Per-sample centering makes the frozen encoder independent of the
        # source cohort's absolute expression scale. The exported compound
        # embedding is the only treatment signal consumed by the response head.
        control_state = self.gene_input_norm(x_control)
        compound_state = F.layer_norm(compound_embedding, (self.gene_dim,))
        gene_state = control_state + compound_state
        treated_gene_z = self.delta_head(self.delta_norm(gene_state))
        pred_gex = self.treated_mean + self.treated_std * treated_gene_z
        pred_delta_gene = pred_gex - x_control
        delta_gene_z = (pred_delta_gene - self.gene_delta_mean) / self.gene_delta_std

        fixed_reactome = self.fixed_reactome(delta_gene_z)
        fixed_hallmark = self.fixed_hallmark(delta_gene_z)
        pred_reactome = torch.exp(self.reactome_log_scale).clamp_max(10.0) * fixed_reactome + self.reactome_bias
        pred_hallmark = torch.exp(self.hallmark_log_scale).clamp_max(10.0) * fixed_hallmark + self.hallmark_bias
        outputs: dict[str, torch.Tensor | None] = {
            "pred_gex": pred_gex,
            "compound_embedding": compound_embedding,
            "pred_pathway": None,
            "pred_delta_gene": pred_delta_gene,
            "pred_delta_gene_z": delta_gene_z,
            "pred_reactome": pred_reactome,
            "pred_hallmark": pred_hallmark,
            "fixed_reactome": fixed_reactome,
            "fixed_hallmark": fixed_hallmark,
        }
        auxiliary: dict[str, torch.Tensor] = {}
        return outputs, auxiliary

    def forward_grouped(
        self,
        x_control: torch.Tensor,
        node_features: torch.Tensor,
        mask: torch.Tensor,
        adjacency: torch.Tensor,
        distance: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
        condition_inverse: torch.Tensor,
        condition_to_smiles: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor]]:
        unique_smiles_embedding = self.molecular_encoder.encode_molecule(
            node_features, mask, adjacency, distance
        )
        if condition_to_smiles is None:
            condition_to_smiles = torch.arange(
                dose.shape[0], device=dose.device, dtype=torch.long
            )
        condition_molecule = unique_smiles_embedding.index_select(0, condition_to_smiles.long())
        unique_condition = self.molecular_encoder.combine_condition(condition_molecule, dose, time)
        unique_compound_embedding = self.project_compound(unique_condition)
        compound_embedding = unique_compound_embedding.index_select(0, condition_inverse.long())
        outputs, auxiliary = self._forward_from_compound(x_control, compound_embedding)
        auxiliary["unique_condition"] = unique_condition
        auxiliary["unique_compound_embedding"] = unique_compound_embedding
        auxiliary["unique_smiles_embedding"] = unique_smiles_embedding
        auxiliary["condition_to_smiles"] = condition_to_smiles
        return outputs, auxiliary

    def forward(
        self,
        x_control: torch.Tensor,
        node_features: torch.Tensor,
        mask: torch.Tensor,
        adjacency: torch.Tensor,
        distance: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        outputs, _ = self._forward_impl(x_control, node_features, mask, adjacency, distance, dose, time)
        return outputs

    def forward_with_aux(
        self,
        x_control: torch.Tensor,
        node_features: torch.Tensor,
        mask: torch.Tensor,
        adjacency: torch.Tensor,
        distance: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor]]:
        return self._forward_impl(x_control, node_features, mask, adjacency, distance, dose, time)

    @classmethod
    def from_files(
        cls,
        genes_file: str | Path,
        pathway_dir: str | Path,
        treated_mean: torch.Tensor,
        treated_std: torch.Tensor,
        **kwargs: Any,
    ) -> "MIPER":
        genes = [line.strip() for line in Path(genes_file).read_text().splitlines() if line.strip()]
        pathway_dir = Path(pathway_dir)
        arrays = np.load(pathway_dir / "pathway_membership.npz", allow_pickle=True)
        stats = json.loads((pathway_dir / "pathway_stats.json").read_text())
        return cls(
            gene_dim=len(genes),
            treated_mean=treated_mean,
            treated_std=treated_std,
            gene_delta_mean=torch.tensor(stats["gene_delta_mean"], dtype=torch.float32),
            gene_delta_std=torch.tensor(stats["gene_delta_std"], dtype=torch.float32),
            reactome_membership=torch.from_numpy(arrays["reactome"].astype(np.float32)),
            hallmark_membership=torch.from_numpy(arrays["hallmark"].astype(np.float32)),
            reactome_raw_mean=torch.tensor(stats["reactome_raw_mean"], dtype=torch.float32),
            reactome_raw_std=torch.tensor(stats["reactome_raw_std"], dtype=torch.float32),
            hallmark_raw_mean=torch.tensor(stats["hallmark_raw_mean"], dtype=torch.float32),
            hallmark_raw_std=torch.tensor(stats["hallmark_raw_std"], dtype=torch.float32),
            **kwargs,
        )


def parameter_summary(model: nn.Module) -> dict[str, int]:
    groups: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        groups[group] = groups.get(group, 0) + parameter.numel()
    groups["total"] = sum(parameter.numel() for parameter in model.parameters())
    groups["trainable"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return groups


def self_test(full_size: bool = False) -> None:
    if full_size:
        model = MIPER(activation_checkpointing=False).eval()
        gene_dim, atom_dim = 978, 28
    else:
        model = MIPER(
            gene_dim=32,
            atom_dim=12,
            molecular_hidden=64,
            molecular_ffn=128,
            condition_dim=96,
            molecular_layers=2,
            molecular_heads=4,
            reactome_dim=7,
            hallmark_dim=3,
            activation_checkpointing=False,
        ).eval()
        gene_dim, atom_dim = 32, 12

    batch, atoms = 2, 9
    node_features = torch.randn(batch, atoms, atom_dim)
    mask = torch.ones(batch, atoms, dtype=torch.bool)
    adjacency = torch.eye(atoms).repeat(batch, 1, 1)
    for index in range(atoms - 1):
        adjacency[:, index, index + 1] = 1.0
        adjacency[:, index + 1, index] = 1.0
    distance = torch.rand(batch, atoms, atoms)
    distance = (distance + distance.transpose(1, 2)) * 0.5
    with torch.inference_mode():
        output = model(
            torch.randn(batch, gene_dim),
            node_features,
            mask,
            adjacency,
            distance,
            torch.rand(batch, 1),
            torch.rand(batch, 1),
        )
    print(json.dumps(parameter_summary(model), indent=2))
    print({name: None if value is None else tuple(value.shape) for name, value in output.items()})
    for name, value in output.items():
        if value is not None and not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite self-test output: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--full-size", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test(full_size=arguments.full_size)
