import numpy as np
from sklearn import ensemble, tree, neural_network, svm


models = {
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    "rf": {
        "model": ensemble.RandomForestRegressor,
        "param": {
            "n_estimators":     lambda: np.random.randint(100, 300+1),
            "max_features":     lambda: np.random.randint(1, 100+1),
            "max_depth":        lambda: np.random.randint(1, 100+1),
            "min_samples_leaf": lambda: np.random.randint(1, 3+1)
        }
    },
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    "gb": {
        "model": ensemble.GradientBoostingRegressor,
        "param": {
            "learning_rate": lambda: np.random.uniform(0.03, 0.3),
            "n_estimators":  lambda: np.random.randint(100, 300+1),
            "max_depth":     lambda: np.random.randint(1, 5+1),
            "max_features":  lambda: np.random.randint(1, 100+1),
            "subsample":     lambda: np.random.uniform(0.1, 1.0),

        }
    },
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
    "ab": {
        "model": ensemble.AdaBoostRegressor,
        "param": {
            "base_estimator": lambda: np.random.choice([
                tree.DecisionTreeRegressor(max_depth=1),
                tree.DecisionTreeRegressor(max_depth=2),
                tree.DecisionTreeRegressor(max_depth=3),
                tree.DecisionTreeRegressor(max_depth=4),
                tree.DecisionTreeRegressor(max_depth=5)
            ]),
            "n_estimators":   lambda: np.random.randint(20, 300+1),
            "learning_rate":  lambda: np.random.uniform(0.1, 2.0),
        }
    },
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    "svm": {
        "model": svm.SVR,
        "param": {
            'max_iter': lambda: 250000,
            "gamma":    lambda: 10**np.random.uniform(-6, -1),
            "C":        lambda: 10**np.random.uniform(-2,  3)
        }
    },
}


class ModelGenerator():

    def __init__(self, model_name, num_iterations=20, random_seed=42):

        from itertools import product
        from copy import copy

        self.model = models[model_name]["model"]
        self.params = models[model_name]["param"]
        self.num_iterations = num_iterations
        self.iteration = 0
        self.random_seed = random_seed

    def __len__(self):
        return self.num_iterations

    def __iter__(self):
        return self

    def __next__(self):

        if self.iteration < self.num_iterations:
            np.random.seed(self.random_seed + self.iteration)
            self.iteration += 1
            self.param = {
                k:v() for (k,v) in self.params.items()
            }
            return self.model(**self.param)
        raise StopIteration
