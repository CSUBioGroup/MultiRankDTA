This is the implementation of MultiRankDTA: A Hybrid Ranking Framework for Robust and Generalizable Drug–Target Affinity Prediction
## 1. Overview
Overview of MultiRankDTA. (a) Base Model and Joint Representation Learning. Drug molecules are transformed into molecular graphs using RDKit and encoded via a multi-layer MPNN to capture structural features. Protein sequences are encoded by the pretrained ESM-2 model, yield contextualized residue embeddings that represent global semantic information. The pooled embeddings are fused into a joint representation and passed through a self-attention module to model cross-modal interactions. This representation is then fed into three ranking experts—pointwise, pairwise, and listwise—to learn numerical affinity values, relative preference, and global ranking structure, respectively. The outputs are integrated via a gating-based mixture-of-experts mechanism to produce the final affinity prediction y. (b) Multi-Source Learning. To enhance generalization across distribution shifts, each sample is independently passed through base models trained on Davis, KIBA, and BindingDB, from which we extract their shared representations and prediction outputs. These vectors are concatenated into a unified high-dimensional representation and fed into a lightweight MetaMLP, improving the predictive performance on the target task.

<p align="center">
  <img width="433" height="178" alt="image" src="https://github.com/user-attachments/assets/5a206855-8743-40a9-ad57-84b729d03e3f" />
</p>


## 2. Project Structure

```text
MultiRankDTA/
├── code/
│   ├── changetransformer.py   # Main model (TransformerModel)
│   ├── create_data.py         # Data preprocessing scripts
│   ├── graph.py               # SMILES to molecular graphs
│   ├── listwise_loss.py       # rank loss functions
│   ├── utils.py               # Metrics and utility functions
│   ├── trynew.py              # multi-Source learning
│   ├── trytaskdavis.py        # Single-dataset training & testing
├── data/
│   ├── davis/
│   ├── kiba/
│   ├── Bindingdb/
│   │   ├── bindingdbIC_2train.csv
│   │   └── bindingdbIC_2test.csv
│   ├── davis_2train.csv
│   ├── davis_2test.csv
│   ├── kiba_2train.csv
│   ├── kiba_2test.csv
└── README.md
```

## 3. Requirements

All experiments were conducted in a Conda environment named `pytorch` on our HPC cluster.

The key dependencies are:

- Python 3.9
- PyTorch 2.5.1
- PyTorch Geometric 2.6.1
- RDKit 2022.9.5 (`rdkit-pypi`)
- scikit-learn 1.5.2
- pandas 2.2.3
- SciPy 1.13.1
- Biopython 1.85
- tqdm 4.66.5
- joblib 1.4.2

A minimal installation example is:

```bash
# create and activate a new environment (you can choose any name)
conda create -n multirankdta python=3.9
conda activate multirankdta

# install PyTorch (please choose the wheel matching your CUDA version)
pip install "torch==2.5.1" "torchaudio" "torchvision"

# install PyTorch Geometric and its dependencies
pip install "torch-geometric==2.6.1"

# chemistry & bioinformatics
pip install "rdkit-pypi==2022.9.5" "biopython==1.85"

# general ML & utilities
pip install "scikit-learn==1.5.2" "pandas==2.2.3" "scipy==1.13.1"
pip install "tqdm==4.66.5" "tensorboard_logger" "joblib==1.4.2"
```

## 4. Datasets

This repository currently provides pre-split CSV files for three benchmark DTA datasets:

```text
data/
├── davis_2train.csv
├── davis_2test.csv
├── kiba_2train.csv
├── kiba_2test.csv
└── Bindingdb/
    ├── bindingdbIC_2train.csv
    └── bindingdbIC_2test.csv
```

Davis

davis_2train.csv, davis_2test.csv

Classical kinase binding affinity dataset (Kd).

KIBA

kiba_2train.csv, kiba_2test.csv

Aggregated KIBA score integrating different bioassays.

BindingDB (IC50 subset)

Bindingdb/bindingdbIC_2train.csv, Bindingdb/bindingdbIC_2test.csv

A curated subset of BindingDB with IC50 measurements.

In addition, we keep the original data format used by earlier DTA works (e.g., DeepDTA / GraphDTA) for reference:
```text
data/
├── davis/
│   ├── Y
│   ├── ligands_can.txt
│   ├── proteins.txt
│   └── folds/
│       ├── train_fold_setting1.txt
│       └── test_fold_setting1.txt
├── kiba/
│   ├── Y
│   ├── ligands_can.txt
│   ├── proteins.txt
│   └── folds/
│       ├── train_fold_setting1.txt
│       └── test_fold_setting1.txt
```

## 5. Training the Base Model (TransformerModel)

The script code/trytaskdavis.py trains the base TransformerModel on a single dataset
with pointwise / pairwise / listwise ranking heads and uncertainty-weighted loss.

5.1 Basic usage

cd code
take the Davis for example
```text
python trytaskdavis.py \
    --datasets davis \
    --device cuda:0 \
    --batch_size 64 \
    --epochs 1000 \
    --save_path multirank/trydavis
```

5.2 Important arguments

--datasets : one of davis, kiba, bindingdbIC

--device : e.g. cuda:0 or cpu

--batch_size : mini-batch size (default: 64)

--epochs : number of training epochs (default: 1000)

--save_path : directory where checkpoints & logs are saved


5.3 Outputs

For each dataset, the script will create a directory, e.g.:

```text
multirank/trydavis/
├── epoch_1.pth
├── epoch_2.pth
├── ...
├── best.pth
├── last.pth
├── all_epochs_metrics.csv
└── all_epochs_predictions.txt
```

epoch_*.pth : checkpoint for each epoch (model + optimizer state).
best.pth : the best model selected by test CI during training.
all_epochs_metrics.csv : per-epoch metrics (e.g., CI, Pearson).
all_epochs_predictions.txt : predicted values for each epoch (optional).


## 6. Multi-Source Learning with MetaMLP

After training base models on Davis / KIBA / BindingDB, we perform multi-source learning
by fusing shared features and pseudo-labels from all three base models.

This is implemented in code/trynew.py.

6.1 Configure paths to base models

At the top of trynew.py, we define the base model checkpoints:

BEST_PTH_FILES = {
    "davismodel": "multirank/trydavis/epoch_891.pth",
    "bindingdbICmodel": "multirank/trybindingdb/epoch_669.pth",
    "kibamodel": "multirank/trykiba/epoch_859.pth"
}


You should update these paths to the checkpoints you actually obtained
(e.g., to your best epochs or to best.pth), for example:

BEST_PTH_FILES = {
    "davismodel":      "multirank/trydavis/best.pth",
    "bindingdbICmodel":"multirank/trybindingdb/best.pth",
    "kibamodel":       "multirank/trykiba/best.pth"
}

6.2 One-shot pipeline: feature extraction + MetaMLP training + evaluation

trynew.py is structured as:

extract_shared_fc() : load base models, extract shared_fc features and pseudo-labels

train_mlp() : train the multi-task MetaMLP with Huber / pairwise / listwise loss and uncertainty weighting

evaluate_mlp() : evaluate CI / Pearson / MSE / R² on each dataset

You can run the full pipeline with a single command:

cd code
python trynew.py


This will sequentially:

For each dataset (davis, bindingdbIC, kiba), run the corresponding base model and save:

```text
multirank/trysix/
├── davis_train_fc.npy,      davis_train_pseudo.npy,      davis_train_real.npy
├── davis_test_fc.npy,       davis_test_pseudo.npy,       davis_test_real.npy
├── bindingdbIC_train_fc.npy, bindingdbIC_train_pseudo.npy, ...
├── kiba_train_fc.npy,       kiba_train_pseudo.npy,       ...
└── ...
```

Train the multi-task MetaMLP on the concatenated features:

Input: [shared_fc features] + [pseudo-label predictions from all source models]

Targets: scaled real labels + scaled pseudo labels for each task

Loss: uncertainty-weighted combination of Huber, pairwise, and ListNet

Per-task and averaged CI, Pearson, RMSE, R²

