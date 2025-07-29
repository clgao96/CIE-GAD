# Correlation Information Enhanced Graph Anomaly Detection via Hypergraph Transformation （CIE-GAD）

This is the official code of the TCYB paper [Correlation Information Enhanced Graph Anomaly Detection via Hypergraph Transformation] (CIE-GAD). 

![The proposed framework](./framework.png)

## Requirments
This code requires the following:
* Python>=3.8
* PyTorch>=1.12.1
* Numpy>=1.24.4
* Scipy>=1.10.1
* Scikit-learn>=1.3.2
* DGL==0.9.0 

## Running the experiments
```
python train.py
```

## Hyperparameters
Here are the training hyperparameters for each dataset:

| Dataset  | lr1  | lr2   | wd1      | wd2     | alpha | a     | b     | drop1 | drop2 | patience |
|----------|------|-------|----------|---------|-------|-------|-------|-------|-------|----------|
| YelpChi  | 0.01 | 0.01  | 0.0005   | 0.0005  | 2     | -0.25 | 2     | 0     | 0     | 150      |
| Amazon   | 0.01 | 0.005 | 0.00005  | 0.001   | 1.5   | -1    | 1.25  | 0.1   | 0.2   | 100      |
| T-Finance| 0.01 | 0.005 | 0.0001   | 0.00005 | 1.5   | -0.75 | -0.5  | 0.1   | 0.2   | 200      |
| Elliptic | 0.01 | 0.05  | 0.001    | 0       | 1     | 0     | 1.75  | 0     | 0.1   | 150      |


## Cite

If you compare with, build on, or use aspects of the CIE-GAD framework, please cite the following:
```
@article{huang2025correlation,
  title={Correlation information enhanced graph anomaly detection via hypergraph transformation},
  author={Huang, Changqin and Gao, Chengling and Li, Ming and Li, Yongzhi and Wang, Xizhe and Jiang, Yunliang and Huang, Xiaodi},
  journal={IEEE Transactions on Cybernetics},
  year={2025},
  volume={55},
  number={6},
  pages={2865-2878},
  publisher={IEEE}
}
```
