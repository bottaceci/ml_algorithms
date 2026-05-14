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

I am then going to use a Jupyter notebook to visualize the results and do the comparisons.

## Project Structure
fill in