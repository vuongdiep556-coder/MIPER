from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class CompactBilinearFusion(nn.Module):
    """Per-sample bilinear interaction between cell and compound features."""

    def __init__(
        self,
        cell_dim: int,
        drug_dim: int,
        out_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, cell_dim, drug_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, cell: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        value = torch.einsum("bc,ocd,bd->bo", cell, self.weight, drug) + self.bias
        return self.dropout(torch.relu(self.norm(value)))


class FlexibleMultiScaleBilinearFusion(nn.Module):
    """V31 multi-scale bilinear fusion with an explicit cell-side width."""

    def __init__(
        self,
        cell_input_dim: int,
        drug_input_dim: int,
        scale_dims: Sequence[int] = (32, 64, 128),
        out_dim: int = 128,
        dropout: float = 0.1,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.cell_proj = nn.ModuleList()
        self.drug_proj = nn.ModuleList()
        self.bilinear = nn.ModuleList()
        for dim in (int(value) for value in scale_dims):
            self.cell_proj.append(
                nn.Sequential(
                    nn.Linear(cell_input_dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            self.drug_proj.append(
                nn.Sequential(
                    nn.Linear(drug_input_dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            self.bilinear.append(
                CompactBilinearFusion(dim, dim, out_dim, dropout)
            )
        self.scale_attention = nn.MultiheadAttention(
            out_dim, attn_heads, dropout=dropout, batch_first=True
        )
        self.scale_score = nn.Linear(out_dim, 1)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(
        self, cell: torch.Tensor, drug: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        tokens = torch.stack(
            [
                bilinear(cell_proj(cell), drug_proj(drug))
                for cell_proj, drug_proj, bilinear in zip(
                    self.cell_proj, self.drug_proj, self.bilinear
                )
            ],
            dim=1,
        )
        if tokens.is_cuda:
            # PyTorch 2.0's fused CUDA SDP backend can emit NaNs for an
            # otherwise finite three-token attention input. The math backend
            # implements the same attention equation and is stable here.
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            ):
                attended, _ = self.scale_attention(
                    tokens, tokens, tokens, need_weights=False
                )
        else:
            attended, _ = self.scale_attention(
                tokens, tokens, tokens, need_weights=False
            )
        weights = torch.softmax(self.scale_score(attended).squeeze(-1), dim=1)
        fused = torch.einsum("bs,bso->bo", weights, attended)
        return {
            "fused": self.out_norm(fused),
            "scale_features": tokens,
            "scale_weights": weights,
        }


class MIPERDrugResponseRegressor(nn.Module):
    """Pathway-guided multiscale response head over a frozen MIPER encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        gex_dim: int = 978,
        finetune_hdim1: int = 512,
        finetune_hdim2: int = 64,
        pathway_hdim: int = 128,
        scale_dims: Sequence[int] = (32, 64, 128),
        bilinear_out: int = 128,
        scale_attn_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = True
        self.scalar = nn.Parameter(torch.ones(1))
        self.bias_scale = nn.Parameter(torch.zeros(1))

        def gene_projection() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(gex_dim, finetune_hdim1),
                nn.BatchNorm1d(finetune_hdim1),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.bgex_proj = gene_projection()
        self.comp_proj = gene_projection()
        self.pathway_proj = nn.Sequential(
            nn.LayerNorm(472 + 49),
            nn.Linear(472 + 49, pathway_hdim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.multiscale = FlexibleMultiScaleBilinearFusion(
            cell_input_dim=gex_dim + pathway_hdim,
            drug_input_dim=gex_dim,
            scale_dims=scale_dims,
            out_dim=bilinear_out,
            dropout=dropout,
            attn_heads=scale_attn_heads,
        )
        head_dim = (
            2 * finetune_hdim1
            + pathway_hdim
            + bilinear_out
        )
        self.pred_head = nn.Sequential(
            nn.Linear(head_dim, finetune_hdim1),
            nn.BatchNorm1d(finetune_hdim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(finetune_hdim1, finetune_hdim2),
            nn.BatchNorm1d(finetune_hdim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(finetune_hdim2, 1),
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True) -> "MIPERDrugResponseRegressor":
        super().train(mode)
        self.encoder.eval()
        return self

    def scaling_layer(self, expression: torch.Tensor) -> torch.Tensor:
        return (expression + self.bias_scale) * self.scalar

    def forward(
        self,
        x_basal: torch.Tensor,
        node_features: torch.Tensor,
        mask: torch.Tensor,
        adjacency: torch.Tensor,
        distance: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoder_out = self.encoder(
            self.scaling_layer(x_basal),
            node_features,
            mask,
            adjacency,
            distance,
            dose,
            time,
        )
        pred_gex = encoder_out["pred_gex"]
        compound = encoder_out["compound_embedding"]
        pathway = torch.cat(
            [encoder_out["pred_reactome"], encoder_out["pred_hallmark"]], dim=1
        )
        pathway_feature = self.pathway_proj(pathway)

        cell = torch.cat([x_basal, pathway_feature], dim=1)
        head_parts = [self.bgex_proj(x_basal), self.comp_proj(compound)]

        multiscale = self.multiscale(cell, compound)
        head_parts.extend([multiscale["fused"], pathway_feature])
        return {
            "pred_ic50": self.pred_head(torch.cat(head_parts, dim=1)),
            "pred_gex": pred_gex,
            "compound_embedding": compound,
            "pred_reactome": encoder_out["pred_reactome"],
            "pred_hallmark": encoder_out["pred_hallmark"],
            "pathway_feature": pathway_feature,
            "scale_features": multiscale["scale_features"],
            "scale_weights": multiscale["scale_weights"],
        }
