# TIDE
Code for paper "XXX"
---

Dependencies
---

python == 3.8.17

pytorch == 1.13.1

PyG (torch-geometric) == 2.2.0

sklearn == 1.3.0

scipy == 1.9.3

numpy == 1.24.3

rdkit == 2023.9.6

Data preparation (Password:1234)
---
For the datasets used in the paper and trained models:

1. The relevant data can be available at the [Link](https://pan.baidu.com/s/1-i8GCBnG1yFElVAg6bviiA?pwd=1234).

2. Unzip the above file to the corresponding directory `./data/`.

3. If you want to train or test the model on different datasets, please modify the parameter settings in the code.

For the DDI database we predicted:

1. The relevant data (~37G / .tar.gz file) can be available at the [Link](https://pan.baidu.com/s/1_igni7S9k65bQlvSC7i1Ww?pwd=1234).

2. Unzip the above file.

Test
---
For _$dataset_ = _$DeepDDI/ZhangDDI/ChChMiner/drugbank/twosides_:

`python test.py --dataset $dataset`

For cold-start datasets _S1_ and _S2_:

`python test.py --dataset drugbank --split cold`

For _MMDDI_ and _DDInter_ datasets:

`python test_MMDDI.py`

For _AUC_FC_ and _AUC_FC_External_ datasets:

`python test_AUC_FC.py`

Train
---
For _$dataset_ = _$DeepDDI/ZhangDDI/ChChMiner/drugbank/twosides_:

`python main.py --dataset $dataset`

For cold-start datasets _S1_ and _S2_:

`python main.py --dataset drugbank --split cold`

For _MMDDI_ and _DDInter_ datasets:

`python main_MMDDI.py`

For _AUC_FC_ and _AUC_FC_External_ datasets:

`python main_AUC_FC.py`


