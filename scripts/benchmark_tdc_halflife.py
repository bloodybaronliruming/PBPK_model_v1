"""Reproduce a locked RDKit2D ExtraTrees model on the official TDC Half_Life_Obach benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from dmpk_toolkit import featurize
from pipeline_common import (base_parser, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stage_output, startup_self_check)


BENCHMARK = 'Half_Life_Obach'
SEEDS = (1, 2, 3, 4, 5)
LOCKED_PARAMS = {
    'n_estimators': 400,
    'min_samples_leaf': 1,
    'max_features': .3,
    'max_depth': 24,
}


def normalize_frame(frame, label):
    frame = frame.copy()
    if 'ID' not in frame and 'Drug_ID' in frame:
        frame = frame.rename(columns={'Drug_ID': 'ID'})
    elif {'ID', 'Drug_ID'} <= set(frame):
        if not frame.ID.equals(frame.Drug_ID):
            raise ValueError(f'TDC {label} 的 ID 与 Drug_ID 列不一致')
        frame = frame.drop(columns='Drug_ID')
    validate_frame(frame, label)
    return frame


def validate_frame(frame, label):
    required = {'ID', 'Drug', 'Y'}
    if not required <= set(frame):
        raise ValueError(f'TDC {label} 缺少列: {sorted(required - set(frame))}')
    if frame.ID.isna().any() or frame.Drug.isna().any() or frame.Y.isna().any():
        raise ValueError(f'TDC {label} 存在空 ID、SMILES 或标签')
    values = frame.Y.to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f'TDC {label} 含非正或非有限半衰期')


def build_model(seed, threads):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('model', ExtraTreesRegressor(random_state=seed, n_jobs=threads, **LOCKED_PARAMS)),
    ])


def fit_predict(train, valid, test, seed, threads):
    combined = pd.concat([train, valid], ignore_index=True)
    all_smiles = pd.concat([combined.Drug, test.Drug], ignore_index=True).tolist()
    features, names = featurize(all_smiles, progress=False, feature_set='rdkit2d')
    x_train = features[:len(combined)]
    x_test = features[len(combined):]
    model = build_model(seed, threads)
    model.fit(x_train, np.log10(combined.Y.to_numpy(float)))
    pred_test = np.power(10., model.predict(x_test))

    # A diagnostic validation model never contributes to the submitted test model.
    train_features = features[:len(train)]
    valid_features = features[len(train):len(combined)]
    diagnostic = build_model(seed, threads)
    diagnostic.fit(train_features, np.log10(train.Y.to_numpy(float)))
    pred_valid = np.power(10., diagnostic.predict(valid_features))
    return model, names, pred_valid, pred_test


def run():
    parser = base_parser('TDC Half_Life_Obach five-seed literature-alignment benchmark')
    parser.add_argument('--tdc-dir', type=Path, default=Path('data/external/tdc_benchmark'))
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--download-only', action='store_true',
                        help='下载或验证 TDC ADMET group，不训练、不读取 TDC test 标签评分')
    args = parser.parse_args()
    configure_logging(args.root, 'benchmark_tdc_halflife')
    if args.threads < 1:
        raise ValueError('--threads 必须为正')
    output = args.output or args.root / 'results/benchmarks/tdc_half_life_obach_rdkit2d_et_v1'
    if not args.download_only:
        startup_self_check(output=output)

    # This is intentionally a separate benchmark track. TDC may download its public
    # benchmark group here; no v15 dataset or v15 test table is opened by this script.
    from tdc.benchmark_group import admet_group
    tdc_dir = args.tdc_dir if args.tdc_dir.is_absolute() else args.root / args.tdc_dir
    group = admet_group(path=str(tdc_dir))
    if args.download_only:
        print(f'TDC ADMET benchmark group ready: {tdc_dir}')
        return
    benchmark = group.get(BENCHMARK)
    train_val = normalize_frame(benchmark['train_val'], 'train_val')
    test = normalize_frame(benchmark['test'], 'test')
    if set(train_val.ID) & set(test.ID):
        raise ValueError('TDC train_val 与 test 存在重复 ID')
    if args.check_only:
        print(f'Validated TDC {BENCHMARK}: train_val={len(train_val)}; test={len(test)}; '
              f'five seeds={SEEDS}; no model trained.')
        return

    test_predictions = []
    rows = []
    with stage_output(output) as out:
        for seed in SEEDS:
            train, valid = group.get_train_valid_split(seed=seed, benchmark=benchmark['name'])
            train = normalize_frame(train, f'train seed={seed}')
            valid = normalize_frame(valid, f'valid seed={seed}')
            if set(train.ID) & set(valid.ID):
                raise ValueError(f'TDC seed={seed} train/valid ID 重叠')
            model, names, valid_prediction, test_prediction = fit_predict(
                train, valid, test, seed, args.threads)
            valid_spearman = float(spearmanr(valid.Y.to_numpy(float), valid_prediction).statistic)
            test_spearman = float(spearmanr(test.Y.to_numpy(float), test_prediction).statistic)
            rows.append({
                'seed': seed, 'train_records': len(train), 'valid_records': len(valid),
                'test_records': len(test), 'validation_spearman': valid_spearman,
                'test_spearman': test_spearman,
            })
            folder = out / f'seed_{seed}'
            folder.mkdir()
            joblib.dump(model, folder / 'model.joblib', compress=3)
            pd.DataFrame({
                'ID': valid.ID, 'smiles': valid.Drug, 'prediction_hours': valid_prediction,
            }).to_csv(folder / 'validation_predictions.csv', index=False)
            # Public benchmark test labels are deliberately not copied into project outputs.
            pd.DataFrame({
                'ID': test.ID, 'smiles': test.Drug, 'prediction_hours': test_prediction,
            }).to_csv(folder / 'tdc_test_predictions.csv', index=False)
            dump_json(folder / 'lineage.json', {
                'benchmark': BENCHMARK, 'seed': seed, 'feature_set': 'rdkit2d',
                'locked_parameters': LOCKED_PARAMS,
                'fit_scope': 'TDC train + validation only; no TDC test labels used for fitting',
            })
            test_predictions.append({benchmark['name']: test_prediction})
        scores = pd.DataFrame(rows)
        aggregate = group.evaluate_many(test_predictions)
        scores.to_csv(out / 'seed_metrics.csv', index=False, float_format='%.6f')
        dump_json(out / 'aggregate_metrics.json', {
            'official_tdc_evaluate_many': aggregate,
            'mean_test_spearman': float(scores.test_spearman.mean()),
            'sd_test_spearman': float(scores.test_spearman.std(ddof=0)),
            'mean_validation_spearman': float(scores.validation_spearman.mean()),
            'seeds': list(SEEDS),
        })
        (out / 'README.md').write_text(
            '# TDC Half_Life_Obach RDKit2D ExtraTrees benchmark\n\n'
            'This separate literature-alignment track uses the official TDC benchmark group, '
            'five TDC seeds and Spearman. The model configuration was locked before reading '
            'TDC test scores. TDC test labels are used only in memory for the official score and '
            'are not exported. This is not independent external validation of v15 because the '
            'underlying Obach source overlaps v15; it is a protocol-aligned literature comparison.\n',
            encoding='utf-8')
        finish_stage(out, 'tdc_half_life_obach_benchmark', inputs={
            'local_tdc_mirror_sha256': sha256(args.root / 'data/external/tdc/half_life_obach.tab'),
            'tdc_train_val_sha256': sha256(tdc_dir / group.name / benchmark['name'] / 'train_val.csv'),
            'tdc_test_sha256': sha256(tdc_dir / group.name / benchmark['name'] / 'test.csv'),
        }, benchmark=BENCHMARK, seeds=list(SEEDS), feature_set='rdkit2d',
        locked_parameters=LOCKED_PARAMS, benchmark_test_evaluated=True,
        v15_test_evaluated=False, external_benchmark_not_independent=True, partial=False)
    print(f'TDC benchmark artifact: {output}')


if __name__ == '__main__':
    run_cli(run)
