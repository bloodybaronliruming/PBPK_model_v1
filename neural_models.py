"""小型 Res-MLP 与有向键消息传递网络；固定 epoch，仅在训练折拟合。"""
from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from tqdm import trange


def atom_features(atom):
    elements = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    number = atom.GetAtomicNum()
    return ([float(number == n) for n in elements] + [float(number not in elements)]
            + [float(min(atom.GetDegree(), 5) == n) for n in range(6)]
            + [atom.GetFormalCharge()/3, float(atom.GetIsAromatic()), atom.GetTotalNumHs()/4, atom.GetMass()/200]
            + [float(atom.GetHybridization() == h) for h in [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]])


ATOM_DIM = 23
BOND_DIM = 6


def molecular_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f'无效分子图: {smiles}')
    atoms = np.asarray([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    src, dst, reverse, bonds = [], [], [], []
    for b in mol.GetBonds():
        vector = [float(b.GetBondType() == typ) for typ in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]]
        vector += [float(b.GetIsConjugated()), float(b.IsInRing())]
        offset = len(src)
        src += [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
        dst += [b.GetEndAtomIdx(), b.GetBeginAtomIdx()]
        reverse += [offset+1, offset]
        bonds += [vector, vector]
    return atoms, np.asarray(bonds, dtype=np.float32).reshape(-1, BOND_DIM), np.asarray(src), np.asarray(dst), np.asarray(reverse)


def collate_graphs(graphs, device):
    atoms, bonds, sources, destinations, reverses, membership = [], [], [], [], [], []
    atom_offset = edge_offset = 0
    for mol_id, (a,b,s,d,r) in enumerate(graphs):
        atoms.append(a); bonds.append(b)
        sources.append(s+atom_offset); destinations.append(d+atom_offset); reverses.append(r+edge_offset)
        membership.extend([mol_id]*len(a))
        atom_offset += len(a); edge_offset += len(b)
    return (torch.as_tensor(np.concatenate(atoms), dtype=torch.float32, device=device),
            torch.as_tensor(np.concatenate(bonds), dtype=torch.float32, device=device),
            *[torch.as_tensor(np.concatenate(x), dtype=torch.long, device=device) for x in [sources, destinations, reverses]],
            torch.as_tensor(membership, dtype=torch.long, device=device), len(graphs))


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width,width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width,width))
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return torch.relu(self.norm(x + self.layers(x)))


class MolecularRegressor(nn.Module):
    def __init__(self, n_features, kind='resmlp', width=128, depth=3, dropout=.15):
        super().__init__()
        self.kind, self.depth, self.width = kind, depth, width
        self.numeric = nn.Sequential(nn.Linear(n_features,width), nn.ReLU(), ResidualBlock(width,dropout), nn.Dropout(dropout))
        if kind == 'dmpnn':
            self.edge_input = nn.Linear(ATOM_DIM+BOND_DIM,width)
            self.edge_update = nn.Linear(width,width, bias=False)
            self.atom_output = nn.Linear(ATOM_DIM+width,width)
        self.head = nn.Sequential(nn.Linear(width*(2 if kind == 'dmpnn' else 1),64), nn.ReLU(), nn.Linear(64,1))

    def forward(self, x, graph=None):
        h = self.numeric(x)
        if self.kind == 'dmpnn':
            atoms,bonds,src,dst,reverse,membership,n_mols = graph
            initial = torch.relu(self.edge_input(torch.cat([atoms[src], bonds],dim=1)))
            edges = initial
            for _ in range(self.depth-1):
                incoming = edges.new_zeros((len(atoms),self.width)).index_add_(0,dst,edges)
                edges = torch.relu(initial + self.edge_update(incoming[src]-edges[reverse]))
            incoming = edges.new_zeros((len(atoms),self.width)).index_add_(0,dst,edges)
            a = torch.relu(self.atom_output(torch.cat([atoms,incoming],dim=1)))
            pooled = a.new_zeros((n_mols,self.width)).index_add_(0,membership,a)
            count = torch.bincount(membership,minlength=n_mols).clamp_min(1).unsqueeze(1)
            h = torch.cat([h, pooled/count],dim=1)
        return self.head(h).squeeze(-1)


class TorchRegressor:
    """只保存 CPU state_dict；重载可在 CPU 推理，不绑定训练 GPU。"""
    def __init__(self, kind='resmlp', width=128, depth=3, dropout=.15, learning_rate=1e-3,
                 epochs=80, batch_size=64, seed=2026, device='cpu', threads=4):
        self.kind, self.width, self.depth, self.dropout = kind,width,depth,dropout
        self.learning_rate,self.epochs,self.batch_size = learning_rate,epochs,batch_size
        self.seed,self.device,self.threads = seed,device,threads

    def network(self):
        return MolecularRegressor(self.n_features_, self.kind,self.width,self.depth,self.dropout)

    def fit(self, X, y, smiles, sample_weight):
        torch.set_num_threads(self.threads)
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
        self.n_features_ = X.shape[1]
        model = self.network().to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate,weight_decay=1e-4)
        graphs = {s:molecular_graph(s) for s in set(smiles)} if self.kind == 'dmpnn' else {}
        rng = np.random.default_rng(self.seed)
        for _ in trange(self.epochs, desc=f'{self.kind} epochs', leave=False):
            model.train()
            order = rng.permutation(len(X))
            for start in range(0,len(X),self.batch_size):
                indices = order[start:start+self.batch_size]
                xx = torch.as_tensor(X[indices],dtype=torch.float32,device=self.device)
                yy = torch.as_tensor(y[indices],dtype=torch.float32,device=self.device)
                ww = torch.as_tensor(sample_weight[indices],dtype=torch.float32,device=self.device)
                graph = collate_graphs([graphs[smiles[i]] for i in indices],self.device) if graphs else None
                optimizer.zero_grad(set_to_none=True)
                losses = torch.nn.functional.huber_loss(model(xx,graph),yy,reduction='none',delta=1.)
                loss = (losses*ww).sum()/ww.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError('神经训练出现非有限损失')
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        self.state_ = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        return self

    def predict(self, X, smiles):
        model = self.network().cpu()
        model.load_state_dict(self.state_); model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0,len(X),self.batch_size):
                xx = torch.as_tensor(X[start:start+self.batch_size],dtype=torch.float32)
                graph = collate_graphs([molecular_graph(s) for s in smiles[start:start+self.batch_size]],'cpu') if self.kind=='dmpnn' else None
                predictions.append(model(xx,graph).numpy())
        return np.concatenate(predictions) if predictions else np.empty(0)
