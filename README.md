# DualTopoDDI
Code for paper "Deciphering Mechanistic Signatures in Drug-Drug Interactions with Dual Topology Graphs"
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

Docker Image
---

We also provide the Dockerfile to build the environment, please refer to the Dockerfile for more details. Make sure you have Docker installed locally, and simply run following command:
   ```shell
   # Build the Docker image
   sudo docker build -t ddi-image:v1 .
   # Create and start the docker container
   sudo docker run --name ddi-con --gpus all -it ddi-image:v1 /bin/bash
   ```

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

DDI inference on your own datasets
---

`$SMILES1` and `$SMILES2` correspond to the SMILES strings of the two drugs respectively, and `$DDI_type` denotes the type of drug-drug interaction to be predicted. When `$DDI_type = DeepDDI`, the model predicts whether an interaction occurs between the two drugs; when `$DDI_type = drugbank`, the model predicts whether the two drugs trigger any of the 86 types of interactions recorded in the DrugBank dataset; when `$DDI_type = twosides`, the model predicts whether the two drugs produce any of the more than 1,000 interactions documented in the TWOSIDES dataset.

`python test_own_data.py --drug1 $SMILES1 --drug2 $SMILES2 --dataset $DDI_type`
