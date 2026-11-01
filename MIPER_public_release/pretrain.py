from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from miper import pretrain as original

ROOT = Path(__file__).resolve().parent
FIELDS = ('gene_rmse', 'gene_pcc', 'delta_gene_pcc', 'reactome_pcc', 'hallmark_pcc')


def read_reference(path: Path) -> dict:
    reference = json.loads(path.read_text(encoding='utf-8'))
    required = {'random_valid', 'drug_ood_valid', 'scaffold_ood_valid'}
    if set(reference['splits']) != required:
        raise ValueError('Selection-reference splits do not match the original protocol')
    groups = [reference['score_baseline']]
    groups.extend(reference['splits'][s] for s in sorted(required))
    groups.append(reference['ood_embedding_rank'])
    for index, group in enumerate(groups):
        keys = FIELDS if index < 4 else ('effective_rank_entropy', 'top10_fraction')
        for key in keys:
            value = group[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f'Unfilled or invalid selection reference: {key}')
    return reference


def required_inputs(cfg: dict) -> list[Path]:
    data = Path(cfg['data']['lincs_dir'])
    pathways = Path(cfg['data']['pathway_dir'])
    paths = [data / n for n in ('conditions.csv', 'genes.txt', 'x_control.npy', 'x_treated.npy', 'mol_features_2d.pkl')]
    paths.extend(pathways / n for n in ('pathway_membership.npz', 'pathway_stats.json', 'reactome_targets.npy', 'hallmark_targets.npy'))
    if not Path(cfg['data']['split_file']).is_file():
        paths.append(data / 'splits.json')
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Portable launcher for unchanged MIPER pretraining')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pretrain.yaml')
    parser.add_argument('--selection-reference', type=Path,
                        help='Original selection references; historical result values are not distributed')
    parser.add_argument('command', choices=('check-inputs', 'prepare', 'smoke', 'train'))
    args = parser.parse_args()
    cfg = original.load_config(args.config)
    missing = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
               for p in required_inputs(cfg) if not p.is_file()]
    if args.command == 'check-inputs':
        print(json.dumps({'missing_inputs': missing,
                          'selection_reference_supplied': args.selection_reference is not None}, indent=2))
        return
    if missing:
        parser.error('Prepared LINCS inputs are required; see docs/PRETRAINING.md: ' + ', '.join(missing))
    if args.command == 'train':
        if args.selection_reference is None:
            parser.error('Original training requires --selection-reference; historical reference metrics are excluded')
        try:
            reference = read_reference(args.selection_reference)
        except (KeyError, ValueError, TypeError) as error:
            parser.error(str(error))
        cfg['selection']['transfer_baseline_file'] = str(args.selection_reference.resolve())
        cfg['selection']['baseline'] = reference['score_baseline']
    if args.command == 'prepare':
        original.prepare_data_index(cfg)
    elif args.command == 'smoke':
        original.run_smoke(cfg)
    else:
        original.run_train(cfg)


if __name__ == '__main__':
    main()
