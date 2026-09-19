"""Fail-fast smoke test for every GPU booster used by the training pipeline."""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithms', default='xgboost,lightgbm,catboost')
    args = parser.parse_args()
    requested = list(dict.fromkeys(args.algorithms.split(',')))
    supported = {'xgboost', 'lightgbm', 'catboost'}
    if not requested or not set(requested) <= supported:
        raise ValueError(f'仅支持: {sorted(supported)}')

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch 看不到 CUDA；停止，禁止静默回退 CPU')
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

    rng = np.random.default_rng(20260913)
    features = rng.normal(size=(256, 16)).astype(np.float32)
    target = (features[:, 0] - .5 * features[:, 1] + rng.normal(scale=.1, size=256)
              ).astype(np.float32)

    if 'xgboost' in requested:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=4, max_depth=2, tree_method='hist',
                             device='cuda', random_state=7)
        model.fit(features, target)
        config = json.loads(model.get_booster().save_config())
        device = config['learner']['generic_param']['device']
        if not str(device).startswith('cuda'):
            raise RuntimeError(f'XGBoost 静默回退: device={device}')
        print(f'xgboost: OK ({device})')

    if 'lightgbm' in requested:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=4, num_leaves=4, device_type='gpu',
                              verbosity=-1, random_state=7)
        model.fit(features, target)
        if model.booster_.params.get('device_type') != 'gpu':
            raise RuntimeError(f'LightGBM 设备异常: {model.booster_.params.get("device_type")}')
        print('lightgbm: OK (gpu)')

    if 'catboost' in requested:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=4, depth=2, task_type='GPU', devices='0',
                                  verbose=False, allow_writing_files=False, random_seed=7)
        model.fit(features, target)
        if model.get_all_params().get('task_type') != 'GPU':
            raise RuntimeError(f'CatBoost 设备异常: {model.get_all_params().get("task_type")}')
        print('catboost: OK (GPU)')


if __name__ == '__main__':
    main()
