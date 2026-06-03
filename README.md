# ML From Scratch

A personal project where I re-implement classical Machine Learning algorithms from scratch using NumPy, validate them against scikit-learn, and study their behavior through experiments, visualizations, and hyperparameter analysis.

The goal is not to build a production-ready ML library, but to understand how these algorithms work internally and gain practical experience with optimization, numerical computing, model evaluation, and software design.

---

## Motivation

Modern ML libraries make it possible to train powerful models with just a few lines of code. While this is extremely useful in practice, it can also hide many of the concepts that make those models work.

This project was created to answer questions such as:

* How does gradient descent actually update model parameters?
* How does a decision tree decide where to split?
* Why does SVM regularization change the decision boundary?
* What does PCA really compute?
* How does backpropagation work inside a neural network?

To answer these questions, each algorithm is implemented from scratch using only NumPy and standard Python tools.

---

## Implemented Algorithms

### Supervised Learning

* Linear Regression
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Perceptron
* Gaussian Naive Bayes
* Multi-Layer Perceptron (MLP)

### Unsupervised Learning

* Principal Component Analysis (PCA)
* K-Means Clustering

---

## Neural Networks

The project includes a fully trainable feed-forward neural network implemented from scratch.

Features:

* Dense (Linear) layers
* Sigmoid activation
* Tanh activation
* Binary Cross Entropy loss
* Categorical Cross Entropy loss
* Softmax output layer
* Binary classification
* Multiclass classification
* Backpropagation
* Xavier initialization
* Layer-based object-oriented architecture

The final implementation follows a simplified deep-learning framework design:

```python
model = Sequential(
    Linear(4, 8),
    Tanh(),
    Linear(8, 3),
    Softmax(),
)
```

---

## Validation Strategy

Every implementation is validated against equivalent models from scikit-learn and, where appropriate, PyTorch.

Typical comparisons include:

* Accuracy
* Precision / Recall / F1
* Confusion matrices
* Decision boundaries
* Loss curves
* Learned parameters
* Hyperparameter sweeps

The goal is to verify both correctness and expected behavior.

---

## Example Experiments

Some of the experiments performed throughout the project include:

* Logistic Regression vs scikit-learn LogisticRegression
* SVM regularization strength and decision boundary analysis
* KNN performance as a function of k
* PCA explained variance and dimensionality reduction
* K-Means clustering and elbow method analysis
* Perceptron on linearly and non-linearly separable datasets
* Gaussian Naive Bayes on datasets that violate feature independence assumptions
* MLP decision boundaries on moons, circles, and multiclass datasets
* Comparison between custom NumPy neural networks and equivalent PyTorch models

---

## Project Structure

```text
.
├── README.md
├── metrics
│   ├── __init__.py
│   ├── classification.py
│   └── regression.py
├── models
│   ├── __init__.py
│   ├── decision_tree.py
│   ├── k_means.py
│   ├── k_means_2.py
│   ├── knn.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── mlp.py
│   ├── mlp_oop.py
│   ├── naive_bayes.py
│   ├── pca.py
│   ├── perceptron.py
│   ├── random_forest.py
│   └── svm.py
├── notebooks
│   ├── 01_algorithm_testing.ipynb
│   ├── 02_linear_regression_testing.ipynb
│   ├── 03_logistic_regression_testing.ipynb
│   ├── 04_decision_tree_testing.ipynb
│   ├── 05_random_forest_testing.ipynb
│   ├── 06_svm_testing.ipynb
│   ├── 07_knn_testing.ipynb
│   ├── 08_pca_testing.ipynb
│   ├── 09_kmeans_testing.ipynb
│   ├── 10_perceptron_testing.ipynb
│   ├── 11_naive_bayes_testing.ipynb
│   ├── 12_mlp_testing.ipynb
│   └── 13_multiclass_testing.ipynb
└── utils
    ├── __init__.py
    ├── encoding.py
    ├── functions.py
    ├── plots.py
    ├── tests.py
    ├── timings.py
    └── torch.py
```

---

## Technologies

* Python
* NumPy
* Matplotlib
* Pandas
* Scikit-learn
* PyTorch
* Jupyter Notebook

---

## Key Takeaways

This project helped me develop a deeper understanding of:

* Numerical optimization
* Gradient-based learning
* Backpropagation
* Statistical learning methods
* Model evaluation
* Software architecture for ML systems
* The similarities and differences between custom implementations and production-grade ML libraries

While the implementations are intentionally simple, they closely mirror the ideas used by modern machine learning frameworks.
