"""Freeze final endpoint candidates before any test metric is calculated."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from dmpk_toolkit import featurize
from pipeline_common import (ROOT, base_parser, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stage_output, startup_self_check, verify_stage)
from train_multitask import MaskedMultiTaskNet


MTL_RUNS = [f'models/multitask/balanced_pappfixed_v3_cuda_seed{seed}' for seed in (2026, 2027, 2028)]
STL = {
    'Papp__human__caco2_ab': ('models/stl/Papp__human__caco2_ab/papp_unitfixed_v3_seed2026', 'processed_v3', 'confirmatory'),
    'CL__human__systemic_iv': ('models/stl/CL__human__systemic_iv/baseline_v1', 'processed_v2', 'confirmatory'),
    'VDss__human__steady_state_iv': ('models/stl/VDss__human__steady_state_iv/baseline_v1', 'processed_v2', 'confirmatory'),
    'Thalf__human__terminal_iv': ('models/stl/Thalf__human__terminal_iv/diagnostic_v1', 'processed_v2', 'exploratory_n2'),
}


def load_mtl_model(run, architecture):
    checkpoint = torch.load(run / architecture / 'model.pt', map_location='cpu', weights_only=False)
    model = MaskedMultiTaskNet(2048 + len(checkpoint['feature_names']), len(checkpoint['tasks']),
                               checkpoint['architecture'], checkpoint['width'], checkpoint['experts'],
                               checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    prep = joblib.load(run / architecture / 'preprocessing.joblib')
    return checkpoint, model, prep


def predict_task(run, architecture, task, smiles):
    checkpoint, model, prep = load_mtl_model(run, architecture)
    col = checkpoint['tasks'].index(task)
    x, _ = featurize(list(smiles), checkpoint['feature_names'], progress=False)
    xx = prep['preprocessor'].transform(x)
    with torch.no_grad():
        z = model(torch.as_tensor(xx, dtype=torch.float32))[:, col].numpy()
    endpoint = prep['task_specs'][task]['endpoint']
    return prep['targets'][col].inverse(z, endpoint)


def verify_reload(run, architecture, task):
    saved = pd.read_csv(run / architecture / 'val_predictions.csv')
    saved = saved.loc[saved.task_id.eq(task)].reset_index(drop=True)
    if saved.empty:
        raise ValueError(f'{run} has no saved validation predictions for {task}')
    actual = predict_task(run, architecture, task, saved.smiles)
    if not np.allclose(actual, saved.predicted_physical, rtol=2e-5, atol=2e-7):
        raise ValueError(f'MTL reload prediction mismatch: {run}')


def ensemble_score(runs, architecture, task, split):
    merged = None
    prediction_columns = []
    for index, run in enumerate(runs):
        frame = pd.read_csv(run / architecture / f'{split}_predictions.csv')
        frame = frame.loc[frame.task_id.eq(task), ['row_id', 'molecule_id', 'observed_physical', 'predicted_physical']]
        name = f'prediction_{index}'
        frame = frame.rename(columns={'predicted_physical': name})
        prediction_columns.append(name)
        merged = frame if merged is None else merged.merge(frame, on=['row_id', 'molecule_id', 'observed_physical'], validate='one_to_one')
    prediction = merged[prediction_columns].mean(axis=1)
    weights = 1 / merged.groupby('molecule_id').molecule_id.transform('size')
    weights = weights / weights.mean()
    return float(np.average(np.abs(merged.observed_physical - prediction), weights=weights))


def run():
    parser = base_parser(__doc__)
    parser.add_argument('--analysis-dir', type=Path,
                        default=ROOT / 'results/analysis/multitask_pappfixed_v3_three_seed_v1')
    args = parser.parse_args()
    configure_logging(args.root, 'freeze_final_candidates')
    output = args.output or args.root / 'models/frozen/final_candidates_v1'
    runs = [args.root / path for path in MTL_RUNS]
    required = [args.analysis_dir / 'complete.json']
    required += [run / 'complete.json' for run in runs]
    required += [args.root / path / 'complete.json' for path, _, _ in STL.values()]
    startup_self_check(required, output=output)
    analysis_meta = verify_stage(args.analysis_dir, 'multitask_three_seed_analysis')
    if analysis_meta.get('test_evaluated') or analysis_meta.get('selection_rule') != 'MTL must beat STL in OOF and validation for all three seeds':
        raise ValueError('three-seed analysis is not an eligible frozen selection source')
    common_inputs = None
    for run_dir in runs:
        meta = verify_stage(run_dir, 'multitask_training')
        if meta.get('test_evaluated') or meta.get('batch_sampler') != 'task_balanced':
            raise ValueError(f'ineligible MTL run: {run_dir}')
        common_inputs = meta['inputs'] if common_inputs is None else common_inputs
        if meta['inputs'] != common_inputs or 'fu__human__plasma' not in meta['tasks']:
            raise ValueError('MTL runs do not share the same frozen inputs/tasks')
        verify_reload(run_dir, 'shared', 'fu__human__plasma')
    candidates = [{
        'task_id': 'fu__human__plasma', 'model_kind': 'multitask_seed_ensemble',
        'architecture': 'shared', 'aggregation': 'arithmetic_mean_physical_scale',
        'run_dirs': [str(run.resolve()) for run in runs], 'data_version': 'processed_v3',
        'inferential_status': 'confirmatory',
        'selection_reason': 'Only endpoint with three-of-three OOF and validation improvements; Shared was the simpler eligible architecture with the lower mean validation MAE.',
        'pretest_ensemble_oof_primary': ensemble_score(runs, 'shared', 'fu__human__plasma', 'train'),
        'pretest_ensemble_validation_primary': ensemble_score(runs, 'shared', 'fu__human__plasma', 'val'),
    }]
    for task, (path, version, status) in STL.items():
        run_dir = args.root / path
        meta = verify_stage(run_dir, 'stl_training')
        selection = json.loads((run_dir / 'selection.json').read_text())
        if meta.get('test_evaluated') or meta.get('excluded_folds'):
            raise ValueError(f'ineligible STL run: {run_dir}')
        candidates.append({'task_id': task, 'model_kind': 'stl', 'algorithm': selection['algorithm'],
                           'run_dir': str(run_dir.resolve()), 'data_version': version,
                           'inferential_status': status,
                           'selection_reason': 'MTL did not meet the pre-registered all-seed OOF-and-validation improvement rule.'})
    registry = {'registry_version': 1, 'test_accessed': False,
                'selection_source': str(args.analysis_dir.resolve()),
                'selection_source_sha256': sha256(args.analysis_dir / 'complete.json'),
                'candidates': candidates,
                'policy': 'No model, architecture, seed, ensemble membership, or hyperparameter may change after test evaluation.'}
    if args.check_only:
        logging.info('Candidate inputs, hashes, selection rule, and MTL reload predictions verified; test remains frozen.')
        return
    with stage_output(output) as out:
        dump_json(out / 'candidate_registry.json', registry)
        lines = ['# Frozen final candidates', '',
                 'This registry was created before test metrics were calculated. The three Shared MLP seeds are averaged on the physical fu scale.', '',
                 '| Task | Frozen model | Data | Status |', '|---|---|---|---|']
        for item in candidates:
            model = 'Shared MLP, seeds 2026/2027/2028 ensemble' if item['model_kind'].startswith('multitask') else f"STL {item['algorithm'].upper()}"
            lines.append(f"| {item['task_id']} | {model} | {item['data_version']} | {item['inferential_status']} |")
        lines += ['', 'Thalf has only two eligible test molecules; its result is descriptive and exploratory.']
        (out / 'selection_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        inputs = {str(path.resolve()): sha256(path / 'complete.json') for path in runs}
        inputs.update({str((args.root / path).resolve()): sha256(args.root / path / 'complete.json') for path, _, _ in STL.values()})
        inputs[str(args.analysis_dir.resolve())] = sha256(args.analysis_dir / 'complete.json')
        finish_stage(out, 'frozen_candidate_registry', inputs=inputs, test_accessed=False,
                     candidate_count=len(candidates), partial=False)
    logging.info('Frozen candidate registry published: %s; test remains unopened by this program.', output)


if __name__ == '__main__':
    run_cli(run)
