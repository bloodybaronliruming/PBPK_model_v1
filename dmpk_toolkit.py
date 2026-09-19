"""第四步：统一特征、数学空间、嵌套固定折训练、指标和可重载模型。"""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from pipeline_common import (ROOT, base_parser, startup_self_check, configure_logging, verify_stage,
    verify_fixed_inputs, stage_output, finish_stage, dump_json, sha256, stable_id, run_cli)
from rebuild_endpoint_datasets import transform_values

ALGORITHMS = ['dummy', 'ridge', 'rf', 'extratrees', 'xgboost', 'lightgbm', 'catboost', 'resmlp', 'dmpnn']
OPTIONAL_MODULES = {'xgboost':'xgboost', 'lightgbm':'lightgbm', 'catboost':'catboost', 'resmlp':'torch', 'dmpnn':'torch'}
GPU_ALGORITHMS = {'xgboost', 'lightgbm', 'catboost', 'resmlp', 'dmpnn'}
FEATURE_SETS = ('ecfp4_rdkit2d', 'ecfp4', 'rdkit2d', 'mechanism2d')
MECHANISM_2D_DESCRIPTORS = (
    'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'HeavyAtomCount', 'NumHeteroatoms',
    'NumHAcceptors', 'NumHDonors', 'NHOHCount', 'NOCount', 'NumRotatableBonds',
    'RingCount', 'NumAromaticRings', 'NumAliphaticRings', 'FractionCSP3',
    'MaxAbsPartialCharge',
)


def read_task_table(path):
    """读取小型端点表而不调用 pandas 的平台相关 CSV C 解析器。

    个别含嵌套活动 ID 列表的人源 F 表会触发某些 pandas 构建的底层崩溃。
    标准库解析保留引号语义，再显式转换任务定义的数值列。
    """
    with Path(path).open(newline='', encoding='utf-8') as stream:
        frame = pd.DataFrame(list(csv.DictReader(stream)))
    numeric = {'raw_value', 'target_value', 'mask', 'assay_id', 'doc_id', 'pH',
               'source_n', 'source_min', 'source_max'} & set(frame.columns)
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame


def load_task(root, datasets, splits, task):
    data_meta = verify_stage(datasets, 'datasets')
    split_meta = verify_stage(splits, 'splits')
    verify_fixed_inputs(data_meta, root)
    verify_fixed_inputs(split_meta, root)
    if split_meta['inputs']['datasets_complete_sha256'] != sha256(datasets / 'complete.json'):
        raise ValueError('fold 清单不是由当前任务数据版本生成')
    registry = json.loads((datasets / 'task_registry.json').read_text())
    if task not in registry:
        raise ValueError(f'未找到合格任务 {task}；检查 task_counts.csv。现有任务: {sorted(registry)}')
    frame = read_task_table(datasets / 'tasks' / f'{task}.csv')
    manifest = pd.read_csv(splits / 'split_manifest.csv', keep_default_na=False)
    if manifest.molecule_id.duplicated().any() or frame.row_id.duplicated().any():
        raise ValueError('分子清单或测定行 ID 不唯一')
    allowed = manifest.eligible.astype(str).str.lower().isin(['true', '1'])
    manifest['eligible'] = allowed
    joined = frame.merge(manifest[['molecule_id','smiles','split','eligible','fold_id','scaffold_group']],
                         on='molecule_id', how='left', suffixes=('', '_manifest'), validate='many_to_one')
    if joined.eligible.isna().any() or not joined.split.eq(joined.split_manifest).all() or not joined.smiles.eq(joined.smiles_manifest).all():
        raise ValueError('任务分子、结构或成员关系与统一清单不一致')
    # 显式检查内容，而不仅依赖前序文件名或 mask。
    spec = registry[task]
    expected = transform_values(joined.raw_value, spec['transform'])
    if not joined['mask'].eq(1).all() or not np.isclose(joined.target_value, expected, rtol=1e-7,atol=1e-7).all():
        raise ValueError('raw/target/mask 不一致')
    if not joined.canonical_unit.eq(spec['unit']).all():
        raise ValueError('任务单位不一致')
    good = joined.loc[joined.eligible].reset_index(drop=True)
    groups = good.groupby('scaffold_group').split.nunique()
    if (groups > 1).any():
        raise ValueError('有效数据存在跨集合骨架交集')
    training = good.loc[good.split.eq('train')]
    if (training.fold_id < 0).any() or (training.groupby('scaffold_group').fold_id.nunique()>1).any():
        raise ValueError('训练骨架与固定 fold 不一致')
    return good, spec, split_meta


def feature_layout(feature_set, descriptor_names=None):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f'未知特征集合: {feature_set}')
    if descriptor_names is not None:
        names = list(descriptor_names)
    elif feature_set == 'ecfp4':
        names = []
    elif feature_set == 'mechanism2d':
        names = list(MECHANISM_2D_DESCRIPTORS)
    else:
        names = [name for name, _ in Descriptors._descList]
    fingerprint = feature_set in {'ecfp4_rdkit2d', 'ecfp4'}
    return fingerprint, names


def featurize(smiles, descriptor_names=None, progress=True, feature_set='ecfp4_rdkit2d'):
    """只计算确定性结构特征；插补、缩放都在 fit_model 的训练范围内。"""
    fingerprint, names = feature_layout(feature_set, descriptor_names)
    funcs = dict(Descriptors._descList)
    if not set(names) <= set(funcs):
        raise ValueError('当前 RDKit 缺少模型使用的描述符，请匹配环境版本')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048) if fingerprint else None
    unique = list(dict.fromkeys(smiles))
    cache = {}
    for smi in tqdm(unique, desc=f'结构特征 {feature_set}', disable=not progress):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms()==0:
            raise ValueError(f'无效 SMILES: {smi}')
        bits = np.zeros(2048,dtype=np.float32)
        if generator is not None:
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), bits)
        values = []
        for name in names:
            try:
                value = float(funcs[name](mol))
                values.append(value if np.isfinite(value) and abs(value)<np.finfo(np.float32).max else np.nan)
            except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
                values.append(np.nan)
        blocks = ([bits] if fingerprint else []) + ([np.asarray(values,dtype=np.float32)] if names else [])
        cache[smi] = np.concatenate(blocks)
    return np.stack([cache[s] for s in smiles]), names


@dataclass
class TargetSpace:
    transform: str
    mean: float = 0.
    std: float = 1.
    z_min: float = -300.
    z_max: float = 300.
    inverse_clip_count: int = 0

    def fit(self, raw, weights):
        z = transform_values(raw,self.transform)
        self.mean = float(np.average(z,weights=weights))
        self.std = max(float(np.sqrt(np.average((z-self.mean)**2,weights=weights))),1e-8)
        # A model may extrapolate far outside the observed transformed range.
        # Keep a generous, data-derived guard for inverse transforms: it prevents
        # 10**z overflow without forcing predictions into a tight empirical box.
        self.z_min = max(-300., float(np.min(z) - 8*self.std))
        self.z_max = min(300., float(np.max(z) + 8*self.std))
        return self

    def forward(self, raw):
        return (transform_values(raw,self.transform)-self.mean)/self.std

    def inverse(self, standardized, endpoint):
        # Several estimators return float32. scipy.expit(float32) saturates to an
        # exact 0/1 around |z| ~= 17, and np.clip then also casts nextafter(1, 0)
        # back to float32 == 1. Convert first so fraction predictions remain in
        # the open interval required by the logit used in saved predictions.
        z = np.asarray(standardized, dtype=np.float64)*self.std+self.mean
        if self.transform == 'log10':
            clipped = np.clip(z, self.z_min, self.z_max)
            self.inverse_clip_count += int(np.count_nonzero(clipped != z))
            z = clipped
        with np.errstate(over='raise',invalid='raise'):
            y = np.power(10.,z) if self.transform=='log10' else expit(z) if self.transform=='logit' else z
        if endpoint=='F':
            y = np.clip(y,0,1)
        if self.transform=='logit':
            y = np.clip(y,np.nextafter(0.,1.),np.nextafter(1.,0.))
        if not np.isfinite(y).all() or (self.transform=='log10' and (y<=0).any()):
            raise FloatingPointError('反变换出现非有限或非正预测')
        return y


@dataclass
class ModelBundle:
    algorithm: str
    estimator: object
    preprocessor: object
    target: TargetSpace
    task_spec: dict
    descriptor_names: list
    train_molecule_ids: list
    train_groups: list
    parameters: dict
    rdkit_version: str
    feature_set: str = 'ecfp4_rdkit2d'

    def predict_features(self, X, smiles):
        xx = self.preprocessor.transform(X)
        pred = self.estimator.predict(xx,smiles) if self.algorithm in {'resmlp','dmpnn'} else self.estimator.predict(xx)
        return self.target.inverse(pred,self.task_spec['endpoint'])

    def predict_smiles(self, smiles):
        if rdBase.rdkitVersion != self.rdkit_version:
            raise ValueError(f'RDKit 版本不匹配: 需要 {self.rdkit_version}，当前 {rdBase.rdkitVersion}')
        x,_ = featurize(smiles, self.descriptor_names,
                        feature_set=getattr(self, 'feature_set', 'ecfp4_rdkit2d'))
        return self.predict_features(x,np.asarray(smiles))


def make_estimator(algorithm, params, seed, threads, device, epochs, batch_size):
    if algorithm=='dummy':
        return DummyRegressor(strategy='mean')
    if algorithm=='ridge':
        # lsqr is stable for the highly collinear ECFP + descriptor matrix and
        # avoids the ill-conditioned normal-equation solve used by auto/cholesky.
        return Ridge(solver='lsqr',tol=1e-6,**params)
    if algorithm in {'rf','extratrees'}:
        cls = RandomForestRegressor if algorithm=='rf' else ExtraTreesRegressor
        return cls(random_state=seed,n_jobs=threads,**params)
    if algorithm=='xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(objective='reg:squarederror', tree_method='hist',
                            device='cuda' if device == 'cuda' else 'cpu',
                            random_state=seed, n_jobs=threads, **params)
    if algorithm=='lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_jobs=threads, verbosity=-1,
                             device_type='gpu' if device == 'cuda' else 'cpu', **params)
    if algorithm=='catboost':
        from catboost import CatBoostRegressor
        gpu = {'task_type': 'GPU', 'devices': '0'} if device == 'cuda' else {'task_type': 'CPU'}
        return CatBoostRegressor(random_seed=seed, thread_count=threads, verbose=False,
                                 allow_writing_files=False, **gpu, **params)
    if algorithm in {'resmlp','dmpnn'}:
        from neural_models import TorchRegressor
        return TorchRegressor(kind=algorithm,seed=seed,threads=threads,device=device,epochs=epochs,batch_size=batch_size,**params)
    raise ValueError(f'不支持算法 {algorithm}')


def candidates(algorithm, trials, seed):
    spaces = {
        'dummy': {}, 'ridge': {'alpha':[.1,1.,10.,100.,1000.]},
        'rf': {'n_estimators':[200,400], 'max_depth':[None,12,24], 'min_samples_leaf':[1,3,5], 'max_features':[.3,.7,1.]},
        'extratrees': {'n_estimators':[200,400], 'max_depth':[None,12,24], 'min_samples_leaf':[1,3,5], 'max_features':[.3,.7,1.]},
        'xgboost': {'n_estimators':[200,400], 'max_depth':[3,5,7], 'learning_rate':[.03,.08], 'subsample':[.8,1.], 'colsample_bytree':[.6,1.]},
        'lightgbm': {'n_estimators':[200,400], 'num_leaves':[15,31], 'learning_rate':[.03,.08], 'min_child_samples':[10,25,50]},
        'catboost': {'iterations':[200,400], 'depth':[4,6], 'learning_rate':[.03,.08], 'l2_leaf_reg':[3,10]},
        'resmlp': {'width':[128,256], 'dropout':[.1,.25], 'learning_rate':[.0003,.001]},
        'dmpnn': {'width':[128,256], 'depth':[3,4], 'dropout':[.1,.25], 'learning_rate':[.0003,.001]},
    }
    if not spaces[algorithm]:
        return [{}]
    size = int(np.prod([len(v) for v in spaces[algorithm].values()]))
    return list(ParameterSampler(spaces[algorithm], n_iter=min(trials,size), random_state=seed))


def molecule_weights(frame):
    counts = frame.groupby('molecule_id').molecule_id.transform('size').to_numpy()
    w = 1./counts
    return w/w.mean()


def fit_model(frame, X, indices, algorithm, parameters, spec, names, args):
    training = frame.iloc[indices]
    weights = molecule_weights(training)
    steps = [('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
    if algorithm in {'ridge','resmlp','dmpnn'}:
        steps += [('scale',StandardScaler())]
    prep = Pipeline(steps)
    xx = prep.fit_transform(X[indices])
    if not np.isfinite(xx).all():
        raise ValueError('训练特征插补后非有限')
    target = TargetSpace(spec['transform']).fit(training.raw_value,weights)
    estimator = make_estimator(algorithm,parameters,args.seed,args.threads,args.device,args.epochs,args.batch_size)
    yy = target.forward(training.raw_value)
    if algorithm in {'resmlp','dmpnn'}:
        estimator.fit(xx,yy,training.smiles.to_numpy(),weights)
    else:
        estimator.fit(xx,yy,sample_weight=weights)
    return ModelBundle(algorithm=algorithm, estimator=estimator, preprocessor=prep, target=target,
                       task_spec=spec, descriptor_names=names,
                       train_molecule_ids=sorted(training.molecule_id.unique()),
                       train_groups=sorted(training.scaffold_group.unique()), parameters=parameters,
                       rdkit_version=rdBase.rdkitVersion, feature_set=args.feature_set)


def primary_score(frame, pred, spec):
    weights = molecule_weights(frame)
    if spec['endpoint'] in {'fu','F'}:
        return float(mean_absolute_error(frame.raw_value,pred,sample_weight=weights))
    return float(np.sqrt(mean_squared_error(transform_values(frame.raw_value,spec['transform']),
                        transform_values(pred,spec['transform']),sample_weight=weights)))


def metrics(frame, pred, spec):
    if len(frame)==0:
        return None
    y = frame.raw_value.to_numpy()
    w = molecule_weights(frame)
    def score(y,p):
        return dict(mae=float(mean_absolute_error(y,p,sample_weight=w)),
                    rmse=float(np.sqrt(mean_squared_error(y,p,sample_weight=w))),
                    r2=float(r2_score(y,p,sample_weight=w)) if len(y)>1 and np.var(y)>0 else None)
    result = dict(records=len(y),molecules=int(frame.molecule_id.nunique()),weighting='equal_total_weight_per_molecule',
                  primary=primary_score(frame,pred,spec),physical=score(y,pred))
    z = transform_values(y,spec['transform']); zp = transform_values(pred,spec['transform'])
    result['transformed'] = score(z,zp)
    ranks = pd.DataFrame({'molecule_id': frame.molecule_id.to_numpy(),
                          'observed': z, 'predicted': zp}).groupby('molecule_id').mean()
    result['spearman_molecule_mean'] = (
        float(spearmanr(ranks.observed, ranks.predicted).statistic)
        if len(ranks) > 1 and ranks.observed.nunique() > 1 and ranks.predicted.nunique() > 1
        else None)
    if spec['endpoint'] in {'fu','F'}:
        result['absolute_within_0.10'] = float(np.average(np.abs(y-pred)<=.1,weights=w))
    pos = (y>0)&(pred>0)
    if pos.any():
        logfe = np.abs(np.log10(pred[pos])-np.log10(y[pos]))
        result['gmfe_positive_subset'] = float(10**np.average(logfe,weights=w[pos]))
        result['within_2fold_positive_subset'] = float(np.average(logfe<=np.log10(2),weights=w[pos]))
        result['within_3fold_positive_subset'] = float(np.average(logfe<=np.log10(3),weights=w[pos]))
    if spec['endpoint']=='fu':
        low = (y<.05)&pos
        result['low_fu_n'] = int(low.sum())
        result['low_fu_log10_rmse'] = float(np.sqrt(np.average((np.log10(y[low])-np.log10(pred[low]))**2,weights=w[low]))) if low.any() else None
    return result


def tune(frame, X, indices, algorithm, options, spec, names, args, records, scope):
    # 所有子折都继承统一 fold；绝不在端点内部重新随机拆分分子。
    if len(options)==1:
        return options[0]
    folds = sorted(frame.iloc[indices].fold_id.unique())
    if len(folds)<2:
        raise ValueError(f'{scope} 无法进行内层 CV')
    scores = []
    for trial, params in enumerate(tqdm(options,desc=f'{algorithm} {scope} 调参',leave=False)):
        prediction = np.full(len(frame),np.nan)
        for fold in folds:
            fit_idx = indices[frame.iloc[indices].fold_id.to_numpy()!=fold]
            val_idx = indices[frame.iloc[indices].fold_id.to_numpy()==fold]
            model = fit_model(frame,X,fit_idx,algorithm,params,spec,names,args)
            prediction[val_idx] = model.predict_features(X[val_idx],frame.iloc[val_idx].smiles.to_numpy())
        value = primary_score(frame.iloc[indices],prediction[indices],spec)
        records.append(dict(algorithm=algorithm,scope=scope,trial=trial,parameters=params,primary=value))
        scores.append(value)
    return options[int(np.argmin(scores))]


def prediction_table(frame, pred, spec, kind, train_groups, include_labels=True):
    cols = ['row_id','molecule_id','smiles','split','fold_id','task_id','scaffold_group']
    out = frame[cols].copy()
    if include_labels:
        out['observed_physical'] = frame.raw_value.to_numpy()
    out['predicted_physical'] = pred
    out['predicted_transformed'] = transform_values(pred,spec['transform'])
    out['canonical_unit'] = spec['unit']
    out['prediction_kind'] = kind
    out['ancestor_train_groups_hash'] = stable_id('\n'.join(train_groups))
    return out


def train_endpoint(endpoint, default_system):
    p = base_parser(f'{endpoint} 单任务训练；固定骨架嵌套 OOF，默认不读取测试标签计算指标')
    p.add_argument('--datasets-dir',type=Path)
    p.add_argument('--splits-dir',type=Path)
    p.add_argument('--species',default='human')
    p.add_argument('--system',default=default_system)
    p.add_argument('--algorithms',default='ridge,rf',help=','.join(ALGORITHMS))
    p.add_argument('--feature-set', choices=FEATURE_SETS, default='ecfp4_rdkit2d',
                   help='结构特征消融；默认 ECFP4 + 全部 RDKit 2D')
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--run-name',default='baseline_v1')
    p.add_argument('--evaluate-test',action='store_true',help='仅在模型方案冻结后显式开启最终 test 评估')
    p.add_argument('--exclude-fold',type=int,action='append',default=[],help='级联的外层保护折；所有拟合排除这些折，可重复')
    args = p.parse_args()
    configure_logging(args.root,f'train_{endpoint}')
    if not args.run_name or Path(args.run_name).name != args.run_name:
        raise ValueError('run-name 必须是单个目录名')
    algorithms = list(dict.fromkeys(args.algorithms.split(',')))
    if not algorithms or not set(algorithms)<=set(ALGORITHMS):
        raise ValueError(f'未知算法: {algorithms}')
    # 常数基线始终保留，选优时不强制神经或复杂模型胜出。
    algorithms = list(dict.fromkeys(['dummy']+algorithms))
    if min(args.trials,args.epochs,args.batch_size,args.threads)<1:
        raise ValueError('trials/epochs/batch-size/threads 必须为正数')
    for algorithm in algorithms:
        if algorithm in OPTIONAL_MODULES:
            importlib.import_module(OPTIONAL_MODULES[algorithm])
    if args.device=='cuda' and set(algorithms)&GPU_ALGORITHMS:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('请求 CUDA，但当前 PyTorch 无可用 GPU')
    task = '__'.join([endpoint,args.species,args.system])
    datasets = args.datasets_dir or args.root/'data/processed_v2/datasets'
    splits = args.splits_dir or args.root/'data/processed_v2/splits'
    output = args.output or args.root/'models/stl'/task/args.run_name
    startup_self_check(output=output)
    frame,spec,split_meta = load_task(args.root,datasets,splits,task)
    if not set(args.exclude_fold)<=set(range(split_meta['folds'])):
        raise ValueError('exclude-fold 不在统一清单中')
    idx = np.flatnonzero(frame.split.eq('train') & ~frame.fold_id.isin(args.exclude_fold))
    folds = sorted(frame.iloc[idx].fold_id.unique())
    if len(folds)<2 or (args.trials>1 and len(folds)<3):
        raise ValueError(f'有效任务仅覆盖 {len(folds)} 个训练折；嵌套调参需至少 3 折，不允许自行重分')
    for f in folds:
        remaining = frame.iloc[idx].loc[frame.iloc[idx].fold_id.ne(f)]
        if remaining.molecule_id.nunique()<2:
            raise ValueError('外层训练分子不足')
    logging.info('task=%s；有效训练记录=%d，分子=%d，folds=%s；算法=%s；特征=%s',
                 task,len(idx),frame.iloc[idx].molecule_id.nunique(),folds,algorithms,args.feature_set)
    if args.check_only:
        logging.info('前序、标签、单位、划分、算法依赖及资源请求检查通过；尚未训练。')
        return
    tuning, summary = [], {}
    with threadpool_limits(limits=args.threads), stage_output(output) as out:
        X,names = featurize(frame.smiles.tolist(), feature_set=args.feature_set)
        fingerprint, _ = feature_layout(args.feature_set, names)
        feature_order = ([f'ECFP4_{i}' for i in range(2048)] if fingerprint else []) + names
        dump_json(out/'feature_schema.json',dict(feature_set=args.feature_set,
                   fingerprint='Morgan radius=2 nbits=2048' if fingerprint else None,
                   descriptor_names=names,feature_count=int(X.shape[1]),rdkit=rdBase.rdkitVersion,
                   feature_order=feature_order,fit_scope='deterministic_structure_only'))
        for algorithm in tqdm(algorithms,desc=f'{task} 算法'):
            folder = out/algorithm;folder.mkdir()
            options = candidates(algorithm,args.trials,args.seed)
            oof_tables = []
            oof = np.full(len(frame),np.nan)
            for fold in tqdm(folds,desc=f'{algorithm} 外层 OOF',leave=False):
                train_idx = idx[frame.iloc[idx].fold_id.to_numpy()!=fold]
                valid_idx = idx[frame.iloc[idx].fold_id.to_numpy()==fold]
                params = tune(frame,X,train_idx,algorithm,options,spec,names,args,tuning,f'outer_{fold}')
                bundle = fit_model(frame,X,train_idx,algorithm,params,spec,names,args)
                if set(bundle.train_groups)&set(frame.iloc[valid_idx].scaffold_group):
                    raise ValueError('外层 OOF 祖先骨架泄漏')
                oof[valid_idx] = bundle.predict_features(X[valid_idx],frame.iloc[valid_idx].smiles.to_numpy())
                fold_dir = folder/f'fold_{fold}';fold_dir.mkdir()
                joblib.dump(bundle,fold_dir/'model.joblib',compress=3)
                dump_json(fold_dir/'lineage.json',dict(train_molecule_ids=bundle.train_molecule_ids,train_groups=bundle.train_groups,
                         heldout_fold=int(fold),additional_excluded_folds=args.exclude_fold,parameters=params))
                oof_tables.append(prediction_table(frame.iloc[valid_idx],oof[valid_idx],spec,'outer_oof',bundle.train_groups))
            if not np.isfinite(oof[idx]).all():
                raise ValueError('OOF 不完整')
            pd.concat(oof_tables,ignore_index=True).to_csv(folder/'oof_predictions.csv',index=False)
            summary[algorithm] = {'oof':metrics(frame.iloc[idx],oof[idx],spec)}
            params = tune(frame,X,idx,algorithm,options,spec,names,args,tuning,'final_training_cv')
            bundle = fit_model(frame,X,idx,algorithm,params,spec,names,args)
            joblib.dump(bundle,folder/'model.joblib',compress=3)
            dump_json(folder/'lineage.json',dict(train_groups=bundle.train_groups,train_molecule_ids=bundle.train_molecule_ids,
                      parameters=params,excluded_folds=args.exclude_fold,task_spec=spec))
            for split in ['val'] + (['test'] if args.evaluate_test else []):
                take = np.flatnonzero(frame.split.eq(split))
                if len(take):
                    prediction = bundle.predict_features(X[take],frame.iloc[take].smiles.to_numpy())
                    prediction_table(frame.iloc[take],prediction,spec,split+'_heldout',bundle.train_groups).to_csv(folder/f'{split}_predictions.csv',index=False)
                    summary[algorithm][split] = metrics(frame.iloc[take],prediction,spec)
            take = np.flatnonzero(frame.split.eq('train') & frame.fold_id.isin(args.exclude_fold))
            if len(take):
                prediction = bundle.predict_features(X[take],frame.iloc[take].smiles.to_numpy())
                prediction_table(frame.iloc[take],prediction,spec,'protected_outer_fold',bundle.train_groups,include_labels=False).to_csv(folder/'excluded_fold_predictions.csv',index=False)
            # 重载验证必须在实际训练产物上通过；不只检查文件存在。
            restored = joblib.load(folder/'model.joblib')
            probe = idx[:min(3,len(idx))]
            if not np.allclose(restored.predict_features(X[probe],frame.iloc[probe].smiles.to_numpy()),
                               bundle.predict_features(X[probe],frame.iloc[probe].smiles.to_numpy())):
                raise ValueError('模型重载预测不一致')
        selected = min(algorithms,key=lambda a:summary[a]['oof']['primary'])
        dump_json(out/'metrics.json',summary)
        dump_json(out/'tuning_history.json',tuning)
        dump_json(out/'selection.json',dict(algorithm=selected,criterion='training_nested_oof_primary',
                  caveat='selected OOF score is selection-biased; assess generalization on heldout validation/test',
                  cascade='use explicit algorithm and protected outer scopes; do not reuse full-train predictions as training priors'))
        versions = {m: getattr(importlib.import_module(m),'__version__','unknown') for m in ['numpy','pandas','sklearn','scipy','joblib']}
        for m in {OPTIONAL_MODULES[a] for a in algorithms if a in OPTIONAL_MODULES}:
            versions[m] = getattr(importlib.import_module(m),'__version__','unknown')
        dump_json(out/'environment.json',versions)
        finish_stage(out,'stl_training',inputs={'datasets_complete_sha256':sha256(datasets/'complete.json'),
                     'splits_complete_sha256':sha256(splits/'complete.json')},task_id=task,excluded_folds=args.exclude_fold,
                     test_evaluated=args.evaluate_test,algorithms=algorithms,seed=args.seed,
                     feature_set=args.feature_set,device=args.device,partial=False)
    logging.info('训练产物: %s；训练 OOF 选定 %s；测试评估=%s',output,selected,args.evaluate_test)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    check = sub.add_parser('check',help='检查新数据与固定划分，训练请使用端点入口')
    check.add_argument('--root',type=Path,default=ROOT)
    check.add_argument('--task',default='fu__human__plasma')
    predict = sub.add_parser('predict',help='已训练模型推理；输出不是训练 OOF 先验')
    predict.add_argument('--model',type=Path,required=True)
    predict.add_argument('--input',type=Path,required=True)
    predict.add_argument('--output',type=Path,required=True)
    evaluate = sub.add_parser('evaluate-test',help='使用已冻结模型评估测试集，不重新训练')
    evaluate.add_argument('--root',type=Path,default=ROOT)
    evaluate.add_argument('--run-dir',type=Path,required=True)
    evaluate.add_argument('--datasets-dir',type=Path)
    evaluate.add_argument('--splits-dir',type=Path)
    evaluate.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.command=='check':
        configure_logging(args.root,'dmpk_toolkit_check')
        frame,_,_ = load_task(args.root,args.root/'data/processed_v2/datasets',args.root/'data/processed_v2/splits',args.task)
        logging.info('数据契约通过：%d 行',len(frame))
    elif args.command=='evaluate-test':
        configure_logging(args.root,'evaluate_test')
        meta = verify_stage(args.run_dir,'stl_training')
        if meta['excluded_folds']:
            raise ValueError('此模型是保护外层折的级联子模型，不能冒充全训练基准。')
        data = args.datasets_dir or args.root/'data/processed_v2/datasets'
        splits = args.splits_dir or args.root/'data/processed_v2/splits'
        frame,spec,_ = load_task(args.root,data,splits,meta['task_id'])
        if meta['inputs'] != {'datasets_complete_sha256':sha256(data/'complete.json'),'splits_complete_sha256':sha256(splits/'complete.json')}:
            raise ValueError('模型与测试数据版本不一致')
        startup_self_check(output=args.output)
        frame = frame.loc[frame.split.eq('test')].reset_index(drop=True)
        if frame.empty:
            raise ValueError('无合格 test 标签')
        selection = json.loads((args.run_dir/'selection.json').read_text())
        algorithm = selection['algorithm']
        bundle = joblib.load(args.run_dir/algorithm/'model.joblib')
        if set(frame.scaffold_group)&set(bundle.train_groups):
            raise ValueError('测试骨架与模型训练范围相交')
        values = bundle.predict_smiles(frame.smiles.tolist())
        with stage_output(args.output) as out:
            prediction_table(frame,values,spec,'frozen_test_evaluation',bundle.train_groups).to_csv(out/'test_predictions.csv',index=False)
            dump_json(out/'metrics.json',dict(algorithm=algorithm,selection_criterion=selection['criterion'],test=metrics(frame,values,spec)))
            finish_stage(out,'test_evaluation',inputs={'frozen_run_sha256':sha256(args.run_dir/'complete.json')},partial=False)
        logging.info('冻结模型测试评估完成，未重新拟合: %s',args.output)
    else:
        startup_self_check([args.model,args.input],output=args.output)
        frame = pd.read_csv(args.input)
        key = 'smiles' if 'smiles' in frame else 'Canonical_SMILES'
        if key not in frame or frame[key].isna().any() or frame.empty:
            raise ValueError('输入需要非空 smiles 或 Canonical_SMILES 列')
        bundle = joblib.load(args.model)
        values = bundle.predict_smiles(frame[key].tolist())
        result = pd.DataFrame({'smiles':frame[key], 'predicted_physical':values,
                  'predicted_transformed':transform_values(values,bundle.task_spec['transform']),
                  'unit':bundle.task_spec['unit'],'prediction_kind':'inference_not_oof'})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        # 独占新文件；不覆盖已有预测。
        with args.output.open('x') as stream:
            result.to_csv(stream,index=False)


if __name__=='__main__':
    run_cli(main)
