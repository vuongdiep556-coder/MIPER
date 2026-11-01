# Pretraining Inputs and Original Selection Protocol

The original model, training loop, objectives, grouped sampler, distributed execution,
checkpoint recovery and validation/selection functions are unchanged.
scripts/pretrain.py is the original entry point; the root pretrain.py adds portable input checks.

## Prepared Inputs

Place these files under data/pretraining/, or change paths in the public configuration:

    lincs_full/
      genes.txt                 # included: ordered 978-gene list
      conditions.csv            # external: aligned conditions and metadata
      x_control.npy             # external: float32, N x 978
      x_treated.npy             # external: float32, N x 978
      mol_features_2d.pkl        # external: source-compatible molecular graph cache
      splits.json               # needed only when regenerating transfer splits
    pathways/
      pathway_membership.npz    # included: memberships and pathway names
      pathway_stats.json        # included: normalization inputs
      reactome_targets.npy      # external: N x 472
      hallmark_targets.npy      # external: N x 49

assets/pretraining/lincs_splits_v3.json is the exact original row-level split.
It requires the original row order; the source loader checks the full row partition.

    python -B pretrain.py check-inputs
    python -B pretrain.py prepare
    python -B pretrain.py smoke

## Original Reference-Based Selection

The training loop computes OOD selection scores relative to reference metrics
and applies eligibility guards. 


    python -B pretrain.py --selection-reference /path/to/selection_reference.json train


Public configuration retains the original architecture, losses, optimizer and stopping settings.

Future training generates its own logs and metrics under runs/pretraining/. No existing

