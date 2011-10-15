import numpy as np

from crfsuite import crfsuite_learn, CRFDataset, CRFDict, CRFTagger

n_samples = 100
n_features = 100
n_instances = 10
n_labels = 10

features_per_sample = 5

data = np.zeros((n_samples, n_features))

indices = np.random.randint(n_features, size=(n_samples, features_per_sample))

data[np.arange(n_samples)[:,None], indices] += 1

labels = np.random.randint(n_labels, size=n_samples)
instances = np.linspace(0, n_samples, n_instances + 1)
instances = np.floor(instances[:-1]).astype(int)

#crf_data = CRFDataset().add_group_from_array(data, labels, instances)
crf_data = CRFDataset().add_groups_from_files('example_files/train_small.txt')

model = crfsuite_learn(crf_data)

crf_data_test = model.get_tagging_data_from_file('example_files/test_small.txt')

tagger = model.get_tagger()

output = tagger.tag(crf_data_test)
