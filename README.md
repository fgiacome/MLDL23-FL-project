# Repository for project 2b group Giacometti, Leo, Licciardi

This repository contains the code of our project.

The repository has three branches. On the main branch, you can find code to do plain FedAvg, on the branch "fed_w_avg" you can find the server which performs our own aggregation strategy, on the branch "self_training" you can find the server and the clients that perform federated domain adaptation.

The training was entirely performed on Google Colab, so
some relevant portions of the code are in the form of
jupyter notebooks.

Please find the notebooks used for training and testing our
models under the directory notebooks/.

To run our trainings, you should replicate the following structure in your Google Drive:

```
MLDL_Datasets/
   GTA5/
   idda/
```

where the folders GTA5 and idda contain the (unpacked) provided datasets.

You can download checkpoints for our models trained on GTA with and without FDA, and for a model trained with the federated domain-adaptation technique, at this link: https://drive.google.com/drive/folders/1_gLQ-sGO60VJfoWXbhkIKhVIwcChHPcG?usp=sharing