from numpy.testing import assert_raises
from crfsuite import CRFDict

KEYS = ['a', 'b', 'c', 'dee', 'eeeeeeee']


def test_init():
    """test initialization of CRFDict object"""
    D = CRFDict()
    assert D.n_items == 0

    D = CRFDict(KEYS)
    assert D.n_items == len(KEYS)


def test_add_keys():
    """test adding keys to CRFDict object"""
    D = CRFDict()

    D.add_keys_batch(KEYS)
    assert D.n_items == len(KEYS)

    for i in range(len(KEYS)):
        D.add_key(KEYS[i])
    assert D.n_items == len(KEYS)


def test_access_keys():
    """test accessing keys and ids of CRFDict object"""
    D = CRFDict(KEYS)

    key_list = D.get_key_list()

    for i in range(len(KEYS)):
        assert key_list[i] == KEYS[i]

        j = D.get_id(KEYS[i])
        assert i == j

        k = D.get_key(i)
        assert k == KEYS[i]


def test_access_keys_error():
    """test ValueErrors raised by CRFDict upon invalid key access"""
    D = CRFDict(KEYS)
    badkey = '_' + ''.join(KEYS)
    assert_raises(ValueError, D.get_id, badkey)
    assert_raises(ValueError, D.get_key, len(KEYS))


if __name__ == '__main__':
    import nose
    nose.runmodule()
