# MIPER: Pretraining Source and Six-Dataset Inference

This independent release candidate contains the complete MIPER pretraining runtime source
and checkpoint inference for TGSA, GDSC1, GDSC2, CCLE, CTRP1 and CTRP2, seeds 41--45.

## Environment and Verification

The verified runtime uses Python 3.10, PyTorch 2.0.1 and NumPy 1.24.4.

## Pretraining

Read docs/PRETRAINING.md for required inputs and commands.

## Layout

- miper/: original model/pretraining source and exact extracted inference utilities.
- configs/: pretraining and inference configuration.
- weights/: one encoder and 30 downstream heads.
- data/six_datasets/: prepared inputs and seed-specific train/valid/test splits.
- data/pretraining/ and assets/pretraining/: small static training inputs.
- 

