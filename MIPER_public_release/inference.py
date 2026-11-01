from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from miper.benchmark_io import BenchmarkDataset, make_loader, metric_values, predict, split_indices
from miper.downstream import MIPERDrugResponseRegressor
from miper.model import MIPER

ROOT = Path(__file__).resolve().parent
DATASETS = ('TGSA', 'GDSC1', 'GDSC2', 'CCLE', 'CTRP1', 'CTRP2')


def load_config(path: str | Path, seed: int) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for section, key in [('data', 'root'), ('data', 'split_root'), ('source', 'encoder_checkpoint')]:
        value = Path(cfg[section][key])
        cfg[section][key] = str(value if value.is_absolute() else ROOT / value)
    cfg['train']['seed'] = seed
    return cfg


def load_model(cfg: dict, dataset: str, seed: int, device: torch.device) -> MIPERDrugResponseRegressor:
    checkpoint = torch.load(cfg['source']['encoder_checkpoint'], map_location='cpu', weights_only=True)
    if checkpoint['architecture'] != 'MIPER-v1' or checkpoint['model_config'] != cfg['encoder']:
        raise ValueError('Encoder checkpoint/config mismatch')
    encoder = MIPER(**cfg['encoder'])
    encoder.load_state_dict(checkpoint['model'], strict=True)
    model = MIPERDrugResponseRegressor(encoder, **cfg['response_head'])
    head = torch.load(ROOT / 'weights' / 'downstream' / dataset / f'seed{seed}.pt',
                      map_location='cpu', weights_only=True)
    if head['dataset'] != dataset or int(head['seed']) != seed:
        raise ValueError('Downstream checkpoint identity mismatch')
    if head['architecture'] != 'MIPER-TOPDR6-v1':
        raise ValueError('Unexpected downstream architecture')
    expected = {k for k in model.state_dict() if not k.startswith('encoder.')}
    if set(head['model']) != expected:
        raise ValueError('Incomplete or unexpected downstream state')
    missing, unexpected = model.load_state_dict(head['model'], strict=False)
    if unexpected or any(not k.startswith('encoder.') for k in missing):
        raise ValueError('Downstream state could not be loaded')
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description='MIPER six-dataset checkpoint inference')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/inference.yaml')
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--seed', choices=range(41, 46), type=int, default=41)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evaluate', action='store_true', help='Compute metrics on selected test rows')
    args = parser.parse_args()
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error('threads and limit must be positive')
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config, args.seed)
    device = torch.device(args.device)
    model = load_model(cfg, args.dataset, args.seed, device)
    data = BenchmarkDataset(cfg, args.dataset)
    rows = split_indices(cfg, args.dataset, 'test')
    if args.limit is not None:
        rows = rows[:args.limit]
    loader = make_loader(data, rows, int(cfg['train']['batch_size']), False)
    with torch.inference_mode():
        row_id, y, prediction = predict(model, loader, device)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, row_id=row_id, y=y, prediction=prediction)
    if args.evaluate:
        print(json.dumps(metric_values(y, prediction), indent=2))
    else:
        print(f'Inference completed: dataset={args.dataset}, seed={args.seed}, rows={len(row_id)}')


if __name__ == '__main__':
    main()
