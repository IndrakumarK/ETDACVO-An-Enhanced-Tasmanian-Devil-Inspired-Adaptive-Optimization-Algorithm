
# 🧠 ETDACVO: Adaptive Evolutionary Optimization for Medical Image Learning

ETDACVO (Enhanced Tasmanian Devil Anti-Conservative Variable Optimization) 
is a hybrid evolutionary optimization framework designed to improve convergence 
stability, cross-domain generalization, and anatomical fidelity in medical image learning systems.

---

## 📌 Overview

Medical imaging models often struggle with:

- Scanner variability  
- Domain shift  
- Limited labeled data  
- Class imbalance  
- Overfitting  
- Unstable convergence  

ETDACVO jointly evolves:

- Data augmentation parameters  
- Optimizer hyperparameters (learning rate, momentum, weight decay)  

using:

- 🐾 Tasmanian Devil Optimization (TDO)  
- 🎯 Anti-Conservative Variable Optimization (ACVO)  
- 📉 EWMA smoothing  

---

## 🚀 Key Results (Reported in Paper)

- +1.0–1.4% accuracy improvement  
- +0.03–0.04 Dice improvement  
- 19–22 fewer epochs to convergence (~30% faster)  
- 45% variance reduction  
- 92.8% cross-domain retention  

---

## 📉 Convergence Definition

Convergence is defined as the first epoch where the training loss drops to  
10% of its initial value, corresponding to a 90% reduction.  

This definition matches Table 10 in the manuscript.

---

## 🔬 Evaluation Protocol Clarification

Validation accuracy is computed strictly on clean validation images  
(i.e., no augmentation is applied during prediction).  

Structural fidelity metrics (SSIM, PSNR, LPIPS, Dice) are computed between  
augmented and original images to evaluate augmentation realism and anatomical preservation.  

Thus, classification evaluation and augmentation fidelity evaluation are separated  
to prevent validation leakage.

---

## ⏱ Computational Transparency

Runtime per evolutionary generation is logged automatically.  
Total evolution runtime is saved and exported to CSV (`experiments/runtime_log.csv`).  

This enables verification of reported computational overhead (~18%).

---

## 📂 Repository Structure

ETDACVO-Medical-Image-Learning/
├── configs/
├── preprocessing/
├── augmentation/
├── optimizer/
├── models/
├── training/
├── experiments/
├── analysis/
├── utils/
├── requirements.txt
├── setup.py
└── README.md

---

## ⚙ Installation

```bash
pip install -r requirements.txt
```

or

```bash
pip install -e .
```

Dependencies include:

- torch
- torchvision
- torchmetrics
- lpips
- scipy
- numpy
- matplotlib

---

## 📚 Citation

```bibtex
@article{indrakumar2026etdacvo,
  title={ETDACVO: An Enhanced Tasmanian-Devil-Inspired Adaptive Optimization Algorithm for Cross-Domain Medical Image Learning},
  author={},
  journal={},
  year={2026}
}
```

---

## 🛡 License

MIT License
