"""Run the registered frozen endpoint models with applicability-domain flags."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from dmpk_toolkit import load_task, transform_values
from freeze_final_candidates import predict_task
from pipeline_common import ROOT, run_cli, startup_self_check


AD_THRESHOLD = 0.30


def fingerprints(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    values = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f'Invalid SMILES: {value}')
        values.append(generator.GetFingerprint(molecule))
    return values


def nearest_train_similarity(query_smiles, train_smiles):
    unique = list(dict.fromkeys(query_smiles))
    query = fingerprints(unique)
    reference = fingerprints(list(dict.fromkeys(train_smiles)))
    if not reference:
        raise ValueError('No training structures are available for applicability-domain calculation')
    scores = dict(zip(unique, [max(DataStructs.BulkTanimotoSimilarity(item, reference)) for item in query]))
    return np.asarray([scores[value] for value in query_smiles])


def candidate_prediction(candidate, smiles):
    unique = list(dict.fromkeys(smiles))
    if candidate['model_kind'] == 'multitask_seed_ensemble':
        members = [predict_task(Path(run), candidate['architecture'], candidate['task_id'], unique)
                   for run in candidate['run_dirs']]
        predicted = np.mean(np.stack(members), axis=0)
    else:
        bundle = joblib.load(Path(candidate['run_dir']) / candidate['algorithm'] / 'model.joblib')
        predicted = bundle.predict_smiles(unique)
    values = dict(zip(unique, predicted))
    return np.asarray([values[value] for value in smiles])


def candidate_training_smiles(root, candidate):
    version = candidate['data_version']
    frame, _, _ = load_task(root, root / f'data/{version}/datasets', root / f'data/{version}/splits',
                            candidate['task_id'])
    if candidate['model_kind'] == 'stl':
        bundle = joblib.load(Path(candidate['run_dir']) / candidate['algorithm'] / 'model.joblib')
        ids = set(bundle.train_molecule_ids)
    else:
        # The three frozen MTL members share a task/data protocol; train is their final fit scope.
        ids = set(frame.loc[frame.split.eq('train'), 'molecule_id'])
    result = frame.loc[frame.molecule_id.isin(ids), 'smiles'].drop_duplicates().tolist()
    if not result:
        raise ValueError(f'Could not recover training structures for {candidate["task_id"]}')
    return result


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--registry', type=Path, default=ROOT / 'models/frozen/final_candidates_v1/candidate_registry.json')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='*', help='Optional registered task IDs to predict')
    args = parser.parse_args()
    startup_self_check([args.registry, args.input])
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    source = pd.read_csv(args.input)
    column = 'smiles' if 'smiles' in source.columns else 'Canonical_SMILES'
    if column not in source or source.empty or source[column].isna().any():
        raise ValueError('Input requires a non-empty smiles or Canonical_SMILES column')
    smiles = source[column].astype(str).tolist()
    registry = json.loads(args.registry.read_text(encoding='utf-8'))
    selected = [item for item in registry['candidates'] if not args.tasks or item['task_id'] in args.tasks]
    unknown = set(args.tasks or []) - {item['task_id'] for item in registry['candidates']}
    if unknown:
        raise ValueError(f'Unregistered tasks requested: {sorted(unknown)}')
    rows = []
    for candidate in selected:
        prediction = candidate_prediction(candidate, smiles)
        similarity = nearest_train_similarity(smiles, candidate_training_smiles(args.root, candidate))
        spec = None
        if candidate['model_kind'] == 'stl':
            bundle = joblib.load(Path(candidate['run_dir']) / candidate['algorithm'] / 'model.joblib')
            spec = bundle.task_spec
        else:
            # All MTL members have the same specification for the frozen task.
            checkpoint = joblib.load(Path(candidate['run_dirs'][0]) / candidate['architecture'] / 'preprocessing.joblib')
            spec = checkpoint['task_specs'][candidate['task_id']]
        rows.append(pd.DataFrame({
            'input_row': np.arange(len(smiles)), 'smiles': smiles, 'task_id': candidate['task_id'],
            'predicted_physical': prediction,
            'predicted_transformed': transform_values(prediction, spec['transform']),
            'unit': spec['unit'], 'inferential_status': candidate['inferential_status'],
            'nearest_train_tanimoto': similarity,
            'applicability_domain': np.where(similarity >= AD_THRESHOLD, 'inside', 'outside'),
            'ad_threshold': AD_THRESHOLD, 'prediction_kind': 'frozen_candidate_inference',
        }))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.output, index=False)


if __name__ == '__main__':
    run_cli(run)
