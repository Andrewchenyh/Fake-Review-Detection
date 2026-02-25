# Fake Review Detection: Traditional ML vs. Deep Learning

## Project Overview
This project focuses on the binary classification of product reviews to distinguish between **Original (OR)** human-written reviews and **Computer-Generated (CG)** fake reviews. As AI-generated content becomes more prevalent, being able to statistically identify synthetic text is crucial for maintaining platform integrity.

## Workflow & Methodology

### 1. Data Analysis & Baselines
* **Dataset**: Analyzed ~40,000 reviews with a perfectly balanced 50/50 split between Real and Fake classes.
* **Baselines**: Implemented **Logistic Regression** and **Support Vector Machines (SVM)** to establish a performance floor.
* **Feature Engineering**: Used TF-IDF vectorization for traditional machine learning models.

### 2. Deep Learning Pipeline (PyTorch)
* **Custom Vocabulary**: Built a robust `Vocabulary` class to handle tokenization and word-to-index mapping.
* **Sequence Optimization**: Conducted statistical analysis on review lengths, setting `max_len=240` to cover the 95th percentile of the data.
* **LSTM Architecture**: Developed a Long Short-Term Memory network to capture sequential dependencies in text.

### 3. Hyperparameter Optimization
Used a **Grid Search** approach to optimize the model’s configuration:
* **Hidden Dimensions**: Tested [64, 128, 256]
* **Learning Rates**: Tested [0.001, 0.0001]
* **Optimizer**: Adam.

## Results
The LSTM model outperformed the baseline models by effectively learning long-range linguistic patterns.

| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| Logistic Regression | 88.0% | 0.89 |
| SVM | 89.0% | 0.89 |
| **LSTM** | **97.0%** | **0.95** |



## Author
**Andrew Chen:** University of California, Davis | Statistics & Economics
