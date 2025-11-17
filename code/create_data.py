aimport json
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from collections import OrderedDict
from rdkit import Chem
from torch_geometric.data import Data
from graph import smile_to_graph, get_atom_feature  
import torch
import esm
import os

# 在文件开头预加载ESM模型
model_path = "/public/home/hpc244711054/perl5/base/graph/tryesm/esm2_t33_650M_UR50D.pt"
esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)


batch_converter = alphabet.get_batch_converter()
esm_model.eval()

def transform_drug(smile):
    # 使用图方法生成特征
    graph_data = smile_to_graph(smile)
    return graph_data


def transform_target(sequence):
    sequence = sequence[:1022]
    

    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()
        esm_model.cuda()
    
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    
    token_representations = results["representations"][33]
    sequence_representation = token_representations[0, 1:-1, :].mean(dim=0)
    
    return sequence_representation.cpu().numpy()


class ListwiseDataset(Dataset):
    def __init__(self, xd=None, xt=None, y=None):
        self.xd = []
        self.xt = []
        self.y = y
        self.original_sequences = xt 
        self.data_len = len(y)

        with torch.no_grad():
            for i in range(self.data_len):
                self.xd.append(transform_drug(xd[i]))
                self.xt.append(transform_target(xt[i]))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        return self.xd[index], self.xt[index], self.y[index], self.original_sequences[index]



def create_csv():
    datasets = ['davis', 'kiba']
    for dataset in datasets:
        fpath = '/public/home/hpc244711054/perl5/base/data/' + dataset + '/'
        train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
        train_fold = [ee for e in train_fold for ee in e]
        valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
        ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
        drugs = []
        prots = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            drugs.append(lg)
        for t in proteins.keys():
            prots.append(proteins[t])
        if dataset in ['davis', 'bindingdb']:

            affinity = [-np.log10(y/1e9) for y in affinity]
        affinity = np.asarray(affinity)
        opts = ['train', 'test']
        for o in opts:
            # 确保路径存在
            output_dir = '/public/home/hpc244711054/perl5/base/graph/tryesm/data/'
            os.makedirs(output_dir, exist_ok=True)
            rows, cols = np.where(np.isnan(affinity) == False)
            if o == 'train':
                rows, cols = rows[train_fold], cols[train_fold]
            elif o == 'test':
                rows, cols = rows[valid_fold], cols[valid_fold]
            with open('/public/home/hpc244711054/perl5/base/graph/tryesm/data/' + dataset + '_2' + o + '.csv', 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [drugs[rows[pair_ind]]]
                    ls += [prots[cols[pair_ind]]]
                    ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                    f.write(','.join(map(str, ls)) + '\n')
        print('\ndataset:', dataset)
        print('train_fold:', len(train_fold))
        print('test_fold:', len(valid_fold))
        print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))


def create_data():
    datasets = ['davis', 'kiba', 'bindingdbIC']
    for dataset in datasets:
        df = pd.read_csv('data/' + dataset + '_2train.csv')
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
        df = pd.read_csv('data/' + dataset + '_2test.csv')
        test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)

        train_data = ListwiseDataset(xd=train_drugs, xt=train_prots, y=train_Y)
        test_data = ListwiseDataset(xd=test_drugs, xt=test_prots, y=test_Y)
        
        # 确保路径存在
        processed_dir = 'data/processed/'
        os.makedirs(processed_dir, exist_ok=True)
        
        train_file_name = 'data/processed/' + dataset + '_additionGRAseventynew_train.pkl'
        test_file_name = 'data/processed/' + dataset + '_additionGRAseventynew_test.pkl'

        with open(train_file_name, 'wb') as f:
            pickle.dump(train_data, f)
            print(dataset + '_additionGRAseventynew_train.pkl processed already')
        with open(test_file_name, 'wb') as f:
            pickle.dump(test_data, f)
            print(dataset + '_additionGRAseventynew_test.pkl processed already')

if __name__ == "__main__":
    create_csv()
    create_data()
