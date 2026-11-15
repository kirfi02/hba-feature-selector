# 🦡 HBA-KNN Metaheuristic Feature Optimization Suite

An interactive wrapper feature selection and benchmarking web application built with **Python**, **Streamlit**, **Scikit-Learn**, and **Mealpy**.

The suite applies the **Honey Badger Algorithm (HBA)** alongside a **K-Nearest Neighbors (KNN)** classifier to eliminate redundant features in high-dimensional tabular datasets, benchmarking HBA head-to-head against **Particle Swarm Optimization (PSO)** and **Genetic Algorithm (GA)**.

## 🔬 Research Background
This software translates swarm-intelligence feature selection research into a reproducible tool, inspired by:
* **Hassan, I. H., Abdullahi, M., Isuwa, J., Yusuf, S. A., & Aliyu, I. T. (2024).** *A comprehensive survey of honey badger optimization algorithm and meta-analysis of its variants and applications.* Franklin Open.
* **Abed, F. N., Sadeghzadeh, M., Hailan, A. M., & Mosleh, M. (2026).** *Honey badger algorithm-optimized morphological segmentation combined with Gabor–KNN classification for meningioma brain tumor.* AIP Advances, 16(11).

## ⚙️ Methodology & Validation Protocol
* **Zero-Leakage Evaluation:** Enforces a strict **75% Training / 25% Unseen Holdout Test** split prior to imputation, scaling, and feature search.
* **Composite Fitness Objective:** Evaluates candidate feature subsets via 5-fold cross-validation on the training split, balancing classification error against feature subset cardinality.
* **Multi-Optimizer Benchmarking:** Compares HBA against PSO and GA under identical population sizes, iteration limits, and data splits across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## 🚀 Local Installation
```bash
git clone https://github.com
cd hba-feature-selector
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m streamlit run app.py
```

If the project folder was moved and the virtual environment reports that `python.exe` cannot be found, recreate it from the project folder with `python -m venv --clear venv`, then rerun the install and launch commands above.

## 👤 Author
**Ukasha M. I. Kirfi**  
B.Sc. Computer Science Alumnus, Federal University of Kashere (FUK)