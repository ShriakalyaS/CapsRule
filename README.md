# CapsRule – Explainable Deep Learning for Classifying Network Attacks

**CapsRule** is a hybrid model combining Capsule Networks and Rule-Based Learning to classify and explain network attacks, especially DDoS intrusions. It is designed to be transparent, efficient, and scalable on large-scale datasets such as CICDDoS2019.

---

## Overview

CapsRule extracts human-understandable rules from a trained capsule network by leveraging dynamic routing coefficients. This method:
- Classifies reflection-based and exploitation-based DDoS attacks
- Extracts range-based rules with high fidelity and comprehensibility
- Enables security-specific decisions with built-in explainability

---

## Dataset

- Dataset used: [CICDDoS2019](https://www.kaggle.com/datasets/dhoogla/cicddos2019)  
- Preprocessing includes feature scaling, label encoding, and attack class filtering.

---

## Model Components

- FFCN (Feed-Forward Capsule Network)
- Dynamic Routing between capsules
- Rule Extraction from routing coefficients

---

## How to Run

Run either of the following:

```bash
jupyter notebook CapsRule.ipynb
# or
python CapsRule.py

## Refernce Paper

This implementation is based on the following IEEE publication.

"CapsRule: Explainable Deep Learning for Classifying Network Attacks"
IEEE Transactions on Neural Networks and Learning Systems, 2024
DOI: 10.1109/TNNLS.2023.3262981
