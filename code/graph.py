import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.rdchem import PeriodicTable
from rdkit.Chem import GetPeriodicTable

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def tensor_dim_slice(tensor, dim, dim_slice):
    return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None),) + (dim_slice,)]

def packshape(shape, dim: int = -1, mask: int = 0b00000001, dtype=torch.uint8, pack=True):
    dim = dim if dim >= 0 else dim + len(shape)
    
    # 设置 bits 值，确保 dtype 合法
    if dtype == torch.uint8:
        bits = 8
    elif dtype == torch.int16:
        bits = 16
    elif dtype == torch.int32:
        bits = 32
    elif dtype == torch.int64:
        bits = 64
    else:
        raise ValueError("Unsupported dtype. Choose one of: torch.uint8, torch.int16, torch.int32, torch.int64.")
    
    if mask == 0b00000001:
        nibble = 1
    elif mask == 0b00000011:
        nibble = 2
    elif mask == 0b00001111:
        nibble = 4
    elif mask == 0b11111111:
        nibble = 8
    else:
        raise ValueError("Unsupported mask value. Choose one of: 0b00000001, 0b00000011, 0b00001111, 0b11111111.")
    
    nibbles = bits // nibble
    if nibbles == 0:
        raise ValueError("Calculated `nibbles` is zero, please check the `mask` and `dtype` values.")
    
    if pack:
        shape = shape[:dim] + (int(np.ceil(shape[dim] / nibbles)),) + shape[dim + 1:]
    else:
        shape = shape[:dim] + (shape[dim] * nibbles,) + shape[dim + 1:]
    
    return shape, nibbles, nibble

def F_unpackbits(tensor, dim: int = -1, mask: int = 0b00000001, shape=None, out=None, dtype=torch.uint8):
    dim = dim if dim >= 0 else dim + tensor.dim()
    shape_, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, dtype=tensor.dtype, pack=False)

    if nibbles == 0:
        raise ValueError("Calculated `nibbles` is zero, please check the `mask` and `dtype` values.")
    
    shape = shape if shape is not None else shape_
    out = out if out is not None else torch.empty(shape, device=tensor.device, dtype=dtype)
    assert out.shape == shape

    if shape[dim] % nibbles == 0:
        shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype=torch.uint8, device=tensor.device)
        shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
        return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out=out)

    else:
        for i in range(nibbles):
            shift = nibble * i
            sliced_output = tensor_dim_slice(out, dim, slice(i, None, nibbles))
            sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
            torch.bitwise_and(sliced_input >> shift, mask, out=sliced_output)
    return out

class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

PACK_NODE_DIM=9
PACK_EDGE_DIM=1
NODE_DIM=PACK_NODE_DIM*8
EDGE_DIM=PACK_EDGE_DIM*8


ATOM_SYMBOL = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
    'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
    'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
    'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
    'Pt', 'Hg', 'Pb', 'Dy', 'Re',  
    'UNK' 
]


def get_atom_feature(atom, mol=None):
    feature = (
        one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOL, allow_unk=True)  
        + one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPE)
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5])
        + one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
        + [atom.IsInRing()]
        + [atom.GetIsAromatic()]
    )
    feature = np.packbits(feature)
    return feature


def one_of_k_encoding(x, allowable_set, allow_unk=True):  
    if x not in allowable_set:
        if allow_unk:
            x = allowable_set[-1]  
        else:
            raise Exception(f'input {x} not in allowable set{allowable_set}!!!')
    return list(map(lambda s: x == s, allowable_set))

HYBRIDIZATION_TYPE = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D
]


def get_bond_feature(bond):
    bond_type = bond.GetBondType()
    feature = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    feature = np.packbits(feature)
    return feature

def smile_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None 

    N = mol.GetNumAtoms()
    node_feature = []
    edge_feature = []
    edge = []

    for i in range(N):
        atom_i = mol.GetAtomWithIdx(i)
        atom_i_features = get_atom_feature(atom_i, mol)
        node_feature.append(atom_i_features)

        for j in range(N):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                edge.append([i, j])
                bond_features_ij = get_bond_feature(bond_ij)
                edge_feature.append(bond_features_ij)

    if len(edge_feature) == 0:
        return None  

    try:
        node_feature = np.stack(node_feature)
        edge_feature = np.stack(edge_feature)
        edge = np.array(edge, dtype=np.uint8)
    except Exception as e:
        return None

    return N, edge, node_feature, edge_feature


def to_pyg_format(N,edge,node_feature,edge_feature):
    graph = Data(
        idx=-1,
        edge_index = torch.from_numpy(edge.T).int(),
        x          = torch.from_numpy(node_feature).byte(),
        edge_attr  = torch.from_numpy(edge_feature).byte(),
    )
    return graph

class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=128, edge_dim=6, aggr='add'):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim + edge_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )
        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        edge_attr = edge_attr[:, :6]  
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=0, reduce=self.aggr)

    def update(self, aggr_out, h):
        if aggr_out.shape[0] < h.shape[0]:
            padding_size = h.shape[0] - aggr_out.shape[0]
            aggr_out = torch.cat([aggr_out, torch.zeros((padding_size, aggr_out.shape[1]), device=aggr_out.device)], dim=0)
        elif aggr_out.shape[0] > h.shape[0]:

            aggr_out = aggr_out[:h.shape[0]]

        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

class MPNNModel(nn.Module):
    def __init__(self, num_layers=3, emb_dim=128, in_dim=70, edge_dim=6, out_dim=1, dropout=0.3):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, emb_dim)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
        self.pool = global_mean_pool
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        h_unpacked = F_unpackbits(data.x, -1).float()[:, :70]
        h = self.lin_in(h_unpacked)  

        for conv in self.convs:
            h = h + conv(h, data.edge_index.long(), F_unpackbits(data.edge_attr, -1).float()[:, :6])  # Trim edge_attr to 6 dimensions
            h = self.dropout(h)  
        h_graph = self.pool(h, data.batch)  
        return h_graph
