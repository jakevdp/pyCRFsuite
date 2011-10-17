import sys

import numpy as np
from scipy.sparse import csr_matrix

from crfsuite import CRFTrainer, CRFDataset

ALGORITHMS = ['lbfgs', 'l2sgd', 'ap', 'pa', 'arow']


def test_algorithms(n_samples=100, n_features=200, n_instances=10):
    X = np.random.random((n_samples, n_features))
    X[np.where(X < 0.8)] = 0
    X = csr_matrix(X)

    labels = np.random.randint(10, size=n_samples)

    instances = np.linspace(0, n_samples, n_samples / n_instances + 1)[:-1]
    instances = np.round(instances)

    data = CRFDataset()
    data.add_group_from_array(X, labels, instances)

    for algorithm in ALGORITHMS:
        trainer = CRFTrainer(algorithm=algorithm, quiet=True)
        trainer.train(data)


if __name__ == '__main__':
    import nose
    nose.runmodule()
