import numpy as np
from .decision_tree import DecisionTree
from collections import Counter

class RandomForest:
    def __init__(self, 
                 n_trees=10,
                 max_depth=100,
                 min_samples_split=2,
                 n_features=None,
                 random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree_seed = self.rng.integers(0, 1_000_000)
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features,
                random_state=tree_seed,
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = self.rng.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        if not self.trees:
            raise ValueError("Model has not been fitted yet.")

        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1) # here also transpose would have worked
        return np.array([self._most_common_label(pred) for pred in tree_preds])
    
