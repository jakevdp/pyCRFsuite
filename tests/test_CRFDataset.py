import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix
from crfsuite import CRFDataset, CRFDict

KEYS1 = ['a', 'b', 'c', 'd']
KEYS2 = ['1', '2', '3']


def test_initialize():
    """Test initialization of CRFDataset object"""
    attrs = CRFDict(KEYS1)
    labels = CRFDict(KEYS2)

    data = CRFDataset(attrs, labels)

    data_attrs = data.get_feature_list()
    for i in range(len(KEYS1)):
        assert KEYS1[i] == data_attrs[i]

    data_labels = data.get_label_list()
    for i in range(len(KEYS2)):
        assert KEYS2[i] == data_labels[i]


def test_matrix_conversion(n_samples=50, n_features=100, n_instances=10):
    """Test conversion of csr matrix to and from CRFDataset"""
    X = np.random.random((n_samples, n_features))
    X[np.where(X < 0.8)] = 0

    X = csr_matrix(X)

    labels = np.random.randint(len(KEYS1), size=n_samples)

    instances = np.linspace(0, n_samples, n_samples / n_instances + 1)[:-1]
    instances = np.round(instances)

    data = CRFDataset()
    data.add_group_from_array(X, labels, instances)

    mat = data.to_matrix()

    assert_array_almost_equal(mat.toarray(), X.toarray())


if __name__ == '__main__':
    import nose
    nose.runmodule()
