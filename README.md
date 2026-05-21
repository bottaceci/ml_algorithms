# ML From Scratch

## Goal
The goal of this project is to replicate the ML algorithms offered by most popular ML libraries from scratch.

## Approach
I am going to write each algorithm, then test it against `sklearn` on the same dataset, and plot failure cases. 
For example, after writing logistic regression from scratch, I am going to compare coefficients, accuracy, confusion matrix, and decision boundary against `sklearn.linear_model.LogisticRegression`.

Each model is going to expose a `scikit-learn`-like API, which allows to do 
```python
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

I am then going to use a Jupyter notebook to visualize the results and do the comparisons. Each Jupyter notebook is going to follow loosely the following structure
1. Goal of the experiment
2. Dataset generation/loading
3. Train/test split
4. Feature scaling
5. Baseline: sklearn model
6. Own model
7. PyTorch model
8. Metrics comparison
9. Plots
10. Hyperparameter sweep
11. Conclusions

## Project Structure
This is the current project structure
```
├── README.md
├── models
│   ├── __init__.py
│   ├── decision_tree.py
│   ├── k_means.py
│   ├── knn.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── naive_bayes.py
│   ├── pca.py
│   ├── perceptron.py
│   ├── random_forest.py
│   └── svm.py
└── notebooks
    └── testing.ipynb
```