# CapsRule – Explainable Deep Learning for Classifying Network Attacks

**CapsRule** is a hybrid model combining **Capsule Networks** and **Rule-based Learning** to classify and explain network attacks, especially DDoS intrusions. It is designed to be transparent, efficient, and scalable on large datasets such as CICDDoS2019.

## Overview

CapsRule extracts human-understandable rules from a trained capsule network by leveraging dynamic routing coefficients. This method:
- Classifies attacks (e.g., reflection- and exploitation-based)
- Extracts range-based rules with high fidelity and comprehensibility
- Supports security-specific decision-making with explainable AI

## Dataset

- CICDDoS2019 dataset is used for training and evaluation.
  '''https://www.kaggle.com/datasets/dhoogla/cicddos2019'''

## Model Components

- FFCN (Feed-Forward Capsule Network)
- Dynamic Routing for capsule agreement
- Rule Extraction based on routing coefficients

##  How to Run

```bash
jupyter notebook CapsRule.ipynb
# or
python CapsRule.py
```


## Reference Paper

This implementation is based on the following IEEE publication. The authors of the paper are not affiliated with this repo.

**"CapsRule: Explainable Deep Learning for Classifying Network Attacks"**  
IEEE Transactions on Neural Networks and Learning Systems, 2024  
DOI: 10.1109/TNNLS.2023.3262981

