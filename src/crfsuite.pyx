"""
Python wrapper for crfsuite functionality.

CRFsuite Data Structure
-----------------------

The crf data structure is as follows:

- A crfsuite dataset is a collection of crfsuite instances
- A crfsuite instance is a collection of crfsuite items,
  each with an integer item label.  The label is known for training data,
  and is learned for test data.
- A crfsuite item is a collection of crfsuite attributes
- A  crfsuite attribute consists of a floating-point value,
  with an integer attribute label.

In addition to this data, a crfsuite dataset contains two crfsuite
dictionaries:

- labels: this gives the string key associated with each integer item label
- attrs: this gives the string key associated with each integer attribute id

Though crfsuite calls these dictionaries, they are more closely related to
python lists than to python dictionaries.

CRFsuite python data interface
------------------------------

Though the data model internal to crfsuite doesn't explicitly use arrays or
matrices, it can be viewed as such for the python interface.  A crfsuite
dataset can be viewed as a sparse matrix of shape (n_samples, n_features).
Each row of the matrix encodes the information in a crfsuite item, and
n_samples is the total number of crfsuite items among all the instances
in the dataset.  Each of the feature dimensions corresponds to an integer
attribute id, with attribute labels given by the associated element in the
attrs dictionary.  The value of feature j for item i is given by data[i, j].
"""

import sys
import os
import time
import warnings
import tempfile

from libc.stdio cimport \
    FILE, fopen, fclose, fprintf, stdout, fflush, fseek, ftell,\
    SEEK_END, SEEK_SET
from libc.stdlib cimport atof, malloc, free
from crfsuite cimport *

from cython.operator import dereference

import numpy as np
cimport numpy as np

from scipy.sparse import csr_matrix

######################################################################
# utility routines & constants
#  used in the interfaces below

FITTYPE_DICT = {'1d': 'crf1d'}

ALGORITHM_DICT = {'lbfgs': 'lbfgs',
                  'l2sgd': 'l2sgd',
                  'ap': 'averaged-perceptron',
                  'averaged-perceptron': 'averaged-perceptron',
                  'pa': 'passive-aggressive',
                  'passive-aggressive': 'passive-aggressive',
                  'arow': 'arow'}


#----------------------------------------------------------------------
# progress function.
#  this is a helper function to print a real-time
#  progress report when reading files
cdef int progress(FILE *fpo, int prev, int current):
    while(prev < current):
        prev += 1
        if prev % 2 == 0:
            if prev % 10 == 0:
                fprintf(fpo, "%d", prev / 10)
                fflush(fpo)
            else:
                fprintf(fpo, ".")
                fflush(fpo)
    return prev


cdef int read_data(crfsuite_data_t* data,
                   FILE* fpi,
                   FILE* fpo,
                   int group):
    """
    This is a cythonized version of the function found in frontend/reader.c
    """
    cdef int n = 0
    cdef int lid = -1
    cdef crfsuite_instance_t inst
    cdef crfsuite_item_t item
    cdef crfsuite_attribute_t cont
    cdef iwa_t* iwa = NULL
    cdef crfsuite_dictionary_t *attrs = data.attrs
    cdef crfsuite_dictionary_t *labels = data.labels
    cdef iwa_token_t *token = NULL
    cdef long filesize = 0, begin = 0, offset = 0
    cdef int prev = 0, current = 0
    cdef char c

    # Initialize the instance
    crfsuite_instance_init(&inst)
    inst.group = group

    # Obtain the file size
    begin = ftell(fpi)
    fseek(fpi, 0, SEEK_END)
    filesize = ftell(fpi) - begin
    fseek(fpi, begin, SEEK_SET)

    #
    fprintf(fpo, "0")
    fflush(fpo)
    prev = 0

    iwa = iwa_reader(fpi)
    while 1:
        token = iwa_read(iwa)
        if(token == NULL):
            break
        # Progress report.
        offset = ftell(fpi)
        current = int((offset - begin) * 100.0 / filesize)
        prev = progress(fpo, prev, current)

        if token.type == IWA_BOI:
            # Initialize an item.
            lid = -1
            crfsuite_item_init(&item)

        elif token.type == IWA_EOI:
            # Append the item to the instance.
            if (lid >= 0):
                crfsuite_instance_append(&inst, &item, lid)
            crfsuite_item_finish(&item)

        elif token.type == IWA_ITEM:
            if (lid == -1):
                lid = labels.get(labels, token.attr)
            else:
                crfsuite_attribute_init(&cont)
                cont.aid = attrs.get(attrs, token.attr)
                if (token.value) and dereference(token.value):
                    cont.value = atof(token.value)
                else:
                    cont.value = 1.0

                crfsuite_item_append_attribute(&item, &cont)

        elif token.type == IWA_NONE or token.type == IWA_EOF:
            # Put the training instance.
            crfsuite_data_append(data, &inst)
            crfsuite_instance_finish(&inst)
            inst.group = group
            n += 1

    progress(fpo, prev, 100)
    fprintf(fpo, "\n")

    return n


cdef int read_data_for_tagging(crfsuite_data_t* data,
                               FILE* fpi,
                               FILE* fpo,
                               int group):
    """
    like read_data, but uses a pre-instantiated attribute and
    label dictionary, ignoring labels and attributes which are
    not already in the dataset
    These are found in data.attrs and data.labels, and should be
    set previous to the function call.
    This is adapted to mimic the behavior in frontend/tag.c, though
    it does not perform the tagging.
    """
    cdef int n = 0
    cdef int lid = -1
    cdef crfsuite_instance_t inst
    cdef crfsuite_item_t item
    cdef crfsuite_attribute_t cont
    cdef iwa_t* iwa = NULL
    cdef crfsuite_dictionary_t *attrs = data.attrs
    cdef crfsuite_dictionary_t *labels = data.labels
    cdef iwa_token_t *token = NULL
    cdef long filesize = 0, begin = 0, offset = 0
    cdef int aid, prev = 0, current = 0
    cdef char c

    cdef int n_labels = labels.num(labels)

    # Initialize the instance
    crfsuite_instance_init(&inst)
    inst.group = group

    # Obtain the file size
    begin = ftell(fpi)
    fseek(fpi, 0, SEEK_END)
    filesize = ftell(fpi) - begin
    fseek(fpi, begin, SEEK_SET)

    #
    fprintf(fpo, "0")
    fflush(fpo)
    prev = 0

    iwa = iwa_reader(fpi)
    while 1:
        token = iwa_read(iwa)
        if(token == NULL):
            break
        # Progress report.
        offset = ftell(fpi)
        current = int((offset - begin) * 100.0 / filesize)
        prev = progress(fpo, prev, current)

        if token.type == IWA_BOI:
            # Initialize an item.
            lid = -1
            crfsuite_item_init(&item)

        elif token.type == IWA_EOI:
            # Append the item to the instance.
            if (lid >= 0):
                crfsuite_instance_append(&inst, &item, lid)
            crfsuite_item_finish(&item)

        elif token.type == IWA_ITEM:
            if (lid == -1):
                # first field in a line is a label name
                lid = labels.to_id(labels, token.attr)
                if lid < 0:
                    lid = n_labels
            else:
                aid = attrs.to_id(attrs, token.attr)
                # If id is in attrs, then associate the attribute
                # with the current item.
                if aid >= 0:
                    if (token.value) and dereference(token.value):
                        crfsuite_attribute_set(&cont, aid, atof(token.value))
                    else:
                        crfsuite_attribute_set(&cont, aid, 1.0)
                    crfsuite_item_append_attribute(&item, &cont)

        elif token.type == IWA_NONE or token.type == IWA_EOF:
            if not crfsuite_instance_empty(&inst):
                crfsuite_data_append(data, &inst)
                inst.group = group
            crfsuite_instance_finish(&inst)
            n += 1

    progress(fpo, prev, 100)
    fprintf(fpo, "\n")

    return n


#------------------------------------------------------------
# callback functions for the trainer
#
cdef int message_callback(void *instance, char *format, va_list args):
    vfprintf(stdout, format, args)
    fflush(stdout)
    return 0

cdef int message_callback_quiet(void *instance, char *format,
                                va_list args):
    return 0


cdef output_tagging_result(FILE *fpo,
                           crfsuite_tagger_t *tagger,
                           crfsuite_instance_t *inst,
                           int *output,
                           crfsuite_dictionary_t *labels,
                           floatval_t score,
                           int output_probability=True,
                           int output_reference=True,
                           int output_marginal=True):
    """
    Utility routine to output the results of a tagging operation in the
    same format as the front-end of the crfsuite
    """
    cdef int i
    cdef floatval_t lognorm, prob, exp_res
    cdef char *label

    if output_probability:
        tagger.lognorm(tagger, &lognorm)
        exp_res = np.exp(score - lognorm)
        fprintf(fpo, "@probability\t%f\n", exp_res)

    for i from 0 <= i < inst.num_items:
        if output_reference:
            labels.to_string(labels, inst.labels[i], &label)
            fprintf(fpo, "%s\t", label)
            labels.free(labels, label)

        labels.to_string(labels, output[i], &label)
        fprintf(fpo, "%s", label)
        labels.free(labels, label)

        if output_marginal:
            tagger.marginal_point(tagger, output[i], i, &prob)
            fprintf(fpo, ":%f", prob)

        fprintf(fpo, "\n")

    fprintf(fpo, "\n")


######################################################################
# crfsuite wrapper classes.
#  these classes wrap give python interfaces to the various structures
#  within crfsuite

cdef class CRFDict(object):
    """Wrapper for crfsuite dictionary

    This is a python class which wraps the dictionary used internally
    in crfsuite.  The crfsuite dictionary is a list of (id, key) pairs,
    where id is an integer, and key is a string.  Though called a
    "dictionary" by crfsuite, the closest python equivalent is a list of
    strings, each with an associated index.

    Parameters
    ----------
    L : iterable of strings
        The list of keys which will make up the dictionary.  The
        id associated with each key is the (zero-indexed) position
        of the key within the list.  If any key is repeated, it will
        be skipped, and the indices shifted.
    """
    cdef crfsuite_dictionary_t* dict

    cdef void set_internal_dict(CRFDict self, crfsuite_dictionary_t* dict):
        """
        C-only function that can be used to wrap an existing crfsuite
        dictionary instance.  The current dictionary instance will be
        dereferenced.
        """
        if dict == NULL:
            raise ValueError("Cannot set NULL dictionary")

        # deallocate dictionary: this duplicates code in __dealloc__
        if self.dict != NULL:
            self.dict.release(self.dict)
            self.dict = NULL
        # ----------------

        self.dict = dict
        self.dict.addref(self.dict)

    def __init__(self, keys=None):
        if self.dict == NULL:
            raise ValueError("Failed to create crfsuite dictionary")
        if keys is not None:
            self.add_keys_batch(keys)

    def __cinit__(self):
        if not crfsuite_create_instance("dictionary", <void**>&self.dict):
            self.dict = NULL

    def __dealloc__(self):
        if self.dict != NULL:
            self.dict.release(self.dict)
            self.dict = NULL

    n_items = property(get_n_items)

    def get_n_items(self):
        """Return the number of items in the dictionary"""
        return self.dict.num(self.dict)

    def add_key(self, key):
        """Add a key to the dictionary.

        If the specified key is already in the dictionary, no change is made.

        Parameters
        ----------
        key : string
            New key to add to the dictionary

        Returns
        -------
        id : integer
            Integer id associated with the key
        """
        return self.dict.get(self.dict, key)

    def add_keys_batch(self, keys):
        """Add a keys to the dictionary.

        Parameters
        ----------
        keys : iterable of strings
            New keys to add to the dictionary

        Returns
        -------
        ids : array of integers, shape = len(keys)
            Integer ids associated with the keys
        """
        cdef int i
        cdef np.ndarray ids = np.zeros(len(keys), dtype=np.int)
        for i from 0 <= i < ids.shape[0]:
            ids[i] = self.dict.get(self.dict, keys[i])
        return ids

    def get_id(self, key):
        """Get id associated with the specified key

        This raises a ValueError if the specified key is not in the
        dictionary.

        Parameters
        ----------
        key : string
            The key to search for in the dictionary

        Returns
        -------
        id : integer
            The id associated with the specified key
        """
        cdef int id = self.dict.to_id(self.dict, key)
        if id < 0:
            raise ValueError("key not in dictionary")
        return id

    def get_key(self, id):
        """Get key associated with the specified id

        This raises a ValueError if the specified id is not in the
        dictionary.

        Parameters
        ----------
        id : integer
            The id to search for in the dictionary

        Returns
        -------
        key : string
            The key associated with the specified key
        """
        cdef char* c = NULL
        self.dict.to_string(self.dict, id, &c)
        if c == NULL:
            raise ValueError("dictionary access out-of-bounds")
        key = str(c)
        self.dict.free(self.dict, c)
        return key

    def get_key_list(self):
        """Get the list of all keys in the dictionary

        Returns
        -------
        keys : list of strings
            The keys in the dictionary, in order of their ids
        """
        cdef int N = self.dict.num(self.dict)
        cdef int i
        cdef char* c
        L = ['' for i in range(N)]
        for i from 0 <= i < N:
            c = NULL
            self.dict.to_string(self.dict, i, &c)
            if c == NULL:
                raise ValueError("dictionary access out-of-bounds")
            L[i] = str(c)
            self.dict.free(self.dict, c)
        return L


cdef class CRFDataset(object):
    """Container for CRF data

    a crfsuite data struct contains an array of instances, and
    two dictionaries: attrs and labels.

    Parameters
    ----------
    attrs : CRFDict object
        attribute dictionary for the dataset.  Attributes are the string ids
        associated with each dimension of the data sample.  If not specified,
        an empty dictionary will be created.

    labels : CRFDict object
        label dictionary for the dataset.  Labels are the string ids associated
        with each item
    """
    cdef readonly int n_labels
    cdef readonly int n_features
    cdef readonly int n_samples
    cdef readonly int n_instances
    cdef readonly int n_groups

    cdef readonly CRFDict attrs
    cdef readonly CRFDict labels

    cdef crfsuite_data_t data

    def __init__(self, attrs=None, labels=None):
        self.n_samples = 0
        self.n_instances = 0
        self.n_labels = self.labels.n_items
        self.n_features = self.attrs.n_items
        self.n_groups = 0

        self.data.attrs = self.attrs.dict
        self.data.labels = self.labels.dict

    def __cinit__(self, attrs=None, labels=None):
        if attrs is None:
            self.attrs = CRFDict()
        else:
            self.attrs = attrs

        if labels is None:
            self.labels = CRFDict()
        else:
            self.labels = labels

        crfsuite_data_init(&self.data)

    def __dealloc__(self):
        crfsuite_data_finish(&self.data)

    def set_attr_dict(self, attrs):
        """Set the attribute dictionary associated with the dataset.

        This will raise an error if the attribute dictionary is already set
        """
        assert self.attrs.n_items == 0
        self.attrs = attrs
        self.data.attrs = self.attrs.dict
        self.n_features = self.attrs.n_items

    def set_label_dict(self, labels):
        """Set the label dictionary associated with the dataset.

        This will raise an error if the attribute dictionary is already set
        """
        assert self.labels.n_items == 0
        self.labels = labels
        self.data.labels = self.labels.dict
        self.n_labels = self.labels.n_items

    #TODO: think about how to handle labels/features with new data groups
    #TODO: change asserts to ValueErrors
    #TODO: add cython types & use csr format to increase speed
    def add_group_from_array(self,
                             data,
                             labels,
                             instances,
                             feature_ids=None,
                             label_ids=None):
        """Add a group of instances from an array

        Parameters
        ----------
        data : array, matrix, or sparse matrix shape=(n_samples, n_features)
            Input data.  data[i, j] represents the weight of the j^th
            feature for item i.

        labels : integer array, shape=(n_samples,)
            the integer labels associated with each item

        instances : integer array, shape=(n_instances,)
            indices giving the starting index of each sequence of indices.
            instance i consists of items in the rows
            data[instances[i]:instances[i + 1]]

        feature_ids : iterable of strings, length = n_features (optional)
            If specified, gives the string ids associated with the features
            If not specified and no previous groups have been added,
            default feature ids are assigned.
            If not specified and previous groups are present, use feature ids
            from previous groups.  In this case, an error is raised if the
            current group has a different number of features.

            an error will be raised if feature_ids is specified and previous
            groups are present [Fix this]

        label_ids : iterable of strings, length = n_labels (optional)
            If specified, gives the string ids associated with the data
            labels.
            If not specified and no previous groups have been added,
            default label ids are assigned.
            If not specified and previous groups are present, use labels
            from previous groups.

            an error will be raised if label_ids is specified and previous
            groups are present [Fix this]
        """
        cdef crfsuite_instance_t inst
        cdef crfsuite_item_t item
        cdef crfsuite_attribute_t attr

        cdef int i, j, jmin, jmax, k, L

        data = csr_matrix(data)
        labels = np.asarray(labels).astype(int)
        instances = np.asarray(instances).astype(int)

        assert len(labels) == data.shape[0]

        #------------------------------------------------------------
        # Take care of feature ids
        if feature_ids is None:
            if self.n_features == 0:
                feature_ids = ["F%i" % (i + 1) for i in xrange(data.shape[1])]
            else:
                feature_ids = self.get_feature_list()
                assert data.shape[1] == self.n_features

        else:
            # TODO: think about this:
            #  if feature_ids are specified and we already have an attr dict
            #  defined, there may be unexpected results.  To be safe, we'll
            #  just raise an error in this case
            assert self.n_features == 0

        assert len(feature_ids) == data.shape[1]
        self.attrs.add_keys_batch(feature_ids)
        self.n_features = len(feature_ids)

        #------------------------------------------------------------
        # Take care of label ids
        if label_ids is None:
            if self.n_labels == 0:
                label_ids = ["L%i" % (i + 1) for i in xrange(labels.max() + 1)]
            else:
                label_ids = self.get_label_list()
        else:
            # TODO: think about this:
            #  if label_ids are specified and we already have a label dict
            #  defined, there may be unexpected results.  To be safe, we'll
            #  just raise an error in this case
            assert self.n_labels == 0

        self.labels.add_keys_batch(label_ids)
        self.n_labels = len(label_ids)

        #------------------------------------------------------------
        # Add to samples and instances count
        self.n_samples += data.shape[0]
        self.n_instances += instances.shape[0]

        #------------------------------------------------------------
        # Populate the data from the python object
        for i from 0 <= i < self.n_instances:
            crfsuite_instance_init(&inst)
            inst.group = self.n_groups

            jmin = instances[i]
            if i < self.n_instances - 1:
                jmax = instances[i + 1]
            else:
                jmax = self.n_samples
            for j from jmin <= j < jmax:
                crfsuite_item_init(&item)
                L = self.data.labels.get(self.data.labels,
                                         label_ids[labels[j]])
                # TODO: this should not loop over all features, but
                # only nonzero features within the csr matrix
                for k from 0 <= k < self.n_features:
                    if data[j, k] == 0:
                        continue
                    crfsuite_attribute_init(&attr)
                    attr.aid = self.data.attrs.get(self.data.attrs,
                                                   feature_ids[k])
                    attr.value = data[j, k]
                    crfsuite_item_append_attribute(&item, &attr)
                # end loop over features

                crfsuite_instance_append(&inst, &item, L)
                crfsuite_item_finish(&item)
            # end loop over items in instance

            crfsuite_data_append(&self.data, &inst)
            crfsuite_instance_finish(&inst)
        # end loop over instances

        self.n_groups += 1

        return self

    def get_feature_list(self):
        return self.attrs.get_key_list()

    def set_feature_list(self, feature_ids):
        assert self.n_features == 0
        self.attrs.add_keys_batch(feature_ids)

    def get_label_list(self):
        return self.labels.get_key_list()

    def set_label_list(self, label_ids):
        assert self.n_labels == 0
        self.labels.add_keys_batch(label_ids)

    def add_groups_from_files(self,
                              training_files,
                              logfile=None):
        """Add groups of instances from an Items With Attributes (IWA) file

        IWA files are the input files used by the crfsuite example scripts.
        This function is designed to facilitate comparison between this
        python wrapper and the examples provided with crfsuite

        Parameters
        ----------
        training_files : filename, or list of filenames
            IWA-formatted files from which groups of instances will be read.
            The data from each file is given its own group ID.  Groups are
            used in cross-validation.

        logfile : filename
            If specified, pipe output to file instead of stdout
        """
        #TODO: think about handling groups/labels/features with mixed input
        #      should we support adding groups from files and arrays within
        #      the same dataset?  This gets complicated.
        cdef char* c_string
        cdef FILE* fp_input

        #------------------------------------------------------------
        # check input parameters
        if type(training_files) == type("string"):
            training_files = [training_files]

        if logfile == None:
            fpo = sys.stdout
        else:
            fpo = open(logfile, 'a')

        #------------------------------------------------------------
        # read data from files
        fpo.write("Reading the data set(s)\n")

        for i, training_file in enumerate(training_files):
            c_string = training_file
            fp_input = fopen(c_string, "r")
            if fp_input == NULL:
                raise ValueError("Failed to open training file %s"
                                 % training_file)

            fpo.write('[%i] %s\n' % (self.n_groups + i, training_file))
            t0 = time.time()

            n = read_data(&self.data,
                           fp_input, PyFile_AsFile(fpo),
                           self.n_groups + i)

            fpo.write('Number of instances: %i\n' % n)
            fpo.write('Seconds required: %.3f\n' % (time.time() - t0))

            fclose(fp_input)

        #------------------------------------------------------------
        # update data statistics
        self.n_samples = crfsuite_data_totalitems(&self.data)
        self.n_features = self.attrs.n_items
        self.n_labels = self.labels.n_items
        self.n_instances = self.data.num_instances
        self.n_groups += len(training_files)

        return self

    def add_groups_from_files_for_tagging(self,
                                          tagging_files,
                                          attr_dict,
                                          label_dict,
                                          logfile=None):
        """add group from an Items With Attributes (IWA) file"""
        #TODO: add documentation
        #TODO: think about handling groups/labels/features with mixed input
        cdef char* c_string
        cdef FILE* fp_input

        assert self.n_features == 0

        #------------------------------------------------------------
        # check input parameters
        if type(tagging_files) == type("string"):
            tagging_files = [tagging_files]

        if logfile == None:
            fpo = sys.stdout
        else:
            fpo = open(logfile, 'a')

        #------------------------------------------------------------
        # Set internal labels & features to those of the training data
        self.set_attr_dict(attr_dict)
        self.set_label_dict(label_dict)

        #------------------------------------------------------------
        # read data from files
        fpo.write("Reading the data set(s)\n")

        for i, tagging_file in enumerate(tagging_files):
            c_string = tagging_file
            fp_input = fopen(c_string, "r")
            if fp_input == NULL:
                raise ValueError("Failed to open tagging file %s"
                                 % tagging_file)

            fpo.write('[%i] %s\n' % (self.n_groups + i, tagging_file))
            t0 = time.time()

            n = read_data_for_tagging(&self.data,
                                       fp_input, PyFile_AsFile(fpo),
                                       self.n_groups + i)

            fpo.write('Number of instances: %i\n' % n)
            fpo.write('Seconds required: %.3f\n' % (time.time() - t0))

            fclose(fp_input)

        #------------------------------------------------------------
        # update data statistics
        self.n_samples = crfsuite_data_totalitems(&self.data)
        self.n_features = self.attrs.n_items
        self.n_labels = self.labels.n_items
        self.n_instances = self.data.num_instances
        self.n_groups += len(tagging_files)

        return self

    def dump(self, fpo=sys.stdout):
        """Report the statistics of the training data."""
        fpo.write("Statistics the data set(s)\n")
        fpo.write("Number of data sets (groups): %d\n" % self.n_groups)
        fpo.write("Number of instances: %d\n" % self.n_instances)
        fpo.write("Number of items: %d\n" % self.n_samples)
        fpo.write("Number of attributes: %d\n" % self.n_features)
        fpo.write("Number of labels: %d\n" % self.n_labels)
        fpo.write("\n")

    def split_data(self, n_groups=1, shuffle=False):
        """Split the data into several groups for cross-validation.

        Parameters
        ----------
        n_groups : integer
            number of groups in which to split the instances
            default = 1

        shuffle : boolean
            if true, shuffle instances in-place before assigning groups.
            warning: if shuffle is set to true, the output of a tagging
            operation will be shuffled is well.
            default = False
        """
        # Shuffle the instances
        if shuffle:
            r = np.random.randint(self.data.num_instances,
                                  size=self.data.num_instances)
            for i from 0 <= i < self.data.num_instances:
                crfsuite_instance_swap(&self.data.instances[i],
                                        &self.data.instances[r[i]])

        # Assign group numbers
        for i from 0 <= i < self.data.num_instances:
            self.data.instances[i].group = (i % n_groups)
        self.n_groups = n_groups

        #TODO : make sure groups count from zero

    def to_matrix(self):
        """Convert dataset to a csr matrix

        Note that this involves a copy of the data.
        """
        cdef crfsuite_instance_t inst
        cdef crfsuite_item_t item
        cdef crfsuite_attribute_t attr

        cdef int i, j, k, n = 0, m = 0
        cdef n_ceil = self.n_samples

        data = np.zeros(n_ceil, dtype='float64')
        idx = np.zeros(n_ceil, dtype=int)
        indptr = np.zeros(self.n_samples + 1, dtype=int)

        for i from 0 <= i < self.data.num_instances:
            inst = self.data.instances[i]
            for j from 0 <= j < inst.num_items:
                item = inst.items[j]
                if n + item.num_contents >= n_ceil:
                    n_ceil *= 2
                    data.resize(n_ceil)
                    idx.resize(n_ceil)
                for k from 0 <= k < item.num_contents:
                    attr = item.contents[k]
                    data[n] = attr.value
                    idx[n] = attr.aid
                    n += 1
                m += 1
                indptr[m] = n

        return csr_matrix((data[:n], idx[:n], indptr),
                          shape=(self.n_samples, self.n_features))

# TODO: think about a way to control the message_callback from
# a python keyword.
cdef class CRFTrainer(object):
    """Wrapper for crfsuite training procedure

    Parameters
    ----------
    fittype: string; default='1d'
        Type of fit to be used.  Currently the only available option is '1d'

    algorithm: string; default='lbfgs'
        Training algorithm to be used.  Options are

        - 'lbfgs': L-BFGS with L1/L2 regularization
        - 'l2sgd': SGD with L2-regularization
        - 'ap': Averaged Perceptron
        - 'pa': Passive Aggressive
        - 'arow': Adaptive Regularization of Weights (AROW)

    quiet: boolean; default=False
        If true, suppress stdout output of training

    other keywords:
        Other keywords are parameters specific to the training algorithm
        TODO: outline parameter options
    """
    cdef crfsuite_trainer_t *trainer

    def __init__(self,
                 fittype="1d",
                 algorithm="lbfgs",
                 quiet=False,
                 **kwargs):
        # Get the fit and algorithm ids
        fit_id = FITTYPE_DICT.get(fittype)
        if fit_id is None:
            raise ValueError("Unknown fit type '%s'" % fittype)

        alg_id = ALGORITHM_DICT.get(algorithm)
        if alg_id is None:
            raise ValueError("Unknown algorithm '%s'" % algorithm)

        # Initialize the trainer instance
        trainer_id = "train/%s/%s" % (fit_id, alg_id)
        if not crfsuite_create_instance(trainer_id, <void**> &self.trainer):
            self.trainer = NULL
            raise ValueError("Failed to create a trainer instance.\n")

        # Set parameters
        cdef crfsuite_params_t *params = NULL
        for (name, value) in kwargs.iteritems():
            params = self.trainer.params(self.trainer)
            if params.set(params, name, value):
                params.release(params)
                raise ValueError("Parameter %s=%s invalid for algorithm %s"
                                 % (name, value, algorithm))
            params.release(params)

        # Set callback procedures that receive messages and taggers
        if quiet:
            self.trainer.set_message_callback(self.trainer, NULL,
                                              &message_callback_quiet)
        else:
            self.trainer.set_message_callback(self.trainer, NULL,
                                              &message_callback)

    def __cinit__(self):
        self.trainer = NULL

    def __dealloc__(self):
        if self.trainer != NULL:
            self.trainer.release(self.trainer)
            self.trainer = NULL

    def train(self, CRFDataset data,
              cross_validate=False,
              holdout=0,
              model_file=None,
              fpo=None):
        """Train the CRFTrainer

        Parameters
        ----------
        data : CRFDataset object
            the training data
        cross_validate : boolean (default=False)
            If true, perform cross-validation.  For this, the dataset should
            be split into multiple groups, either through loading of multiple
            files or by calling data.split_data(n_groups).  If cross-validate
            is true, no model will be returned.
        holdout : int (default=0)
            For holdout=M, use the M-th group for holdout evaluation and the
            rest for training
        model_file : string (default=None)
            location to save the model.  If none, the model will be saved to
            a temporary file.
        fpo : file stream or file name
            the python file stream for output of training progress.  If not
            specified, output will be directed to stdout.

        Returns
        -------
        model : a CRFModel object

        Notes
        -----
        As currently written, crfsuite is designed for input and output only
        via file.  That is, there is no way to create a model and store it
        in memory without first saving it in a temporary file.  This
        deficiency of the library could be fairly easily corrected with
        an upstream patch which would allow the user to pass a file stream
        rather than a file name to the training algorithm.
        """
        if model_file is None or model_file == "":
            model_file = tempfile.mktemp()
            remove_tmp_file = True
        else:
            remove_tmp_file = False

        cdef int i

        if fpo is None:
            fpo = sys.stdout
        elif isinstance(fpo, file):
            pass
        else:
            fpo = open(fpo, 'w')

        if cross_validate:
            for i from 0 <= i < data.n_groups:
                fpo.write("===== Cross validation (%d/%d) =====\n"
                          % (i+1, data.n_groups))
                if self.trainer.train(self.trainer, &data.data, "", i):
                    raise ValueError("training failed")
                fpo.write('\n')

            return None

        else:
            if self.trainer.train(self.trainer, &data.data,
                                  model_file, holdout):
                raise ValueError("training failed")

            model = CRFModel(model_file)
            if remove_tmp_file:
                os.remove(model_file)
            return model


cdef class CRFModel(object):
    """Python wrapper for a crfsuite model

    Parameters
    ----------
    model_file : string
        The model file resulting from CRFTrainer.train().  The CRFModel
        instance will be created from this file
    """
    cdef crfsuite_model_t *model

    def __init__(self, model_file):
        if crfsuite_create_instance_from_file(model_file,
                                              <void**>&self.model):
            raise ValueError("Could not create model")

    def __cinit__(self):
        self.model = NULL

    def __dealloc__(self):
        if self.model != NULL:
            self.model.release(self.model)
            self.model = NULL

    def get_label_dict(self):
        """
        Return the CRFDict object which holds the labels for the model
        """
        cdef crfsuite_dictionary_t *labels
        cdef CRFDict label_dict = CRFDict()
        if self.model.get_labels(self.model, &labels):
            raise ValueError("Could not create dictionary")
        label_dict.set_internal_dict(labels)
        labels.release(labels)
        return label_dict

    def get_attr_dict(self):
        """
        Return the CRFDict object which holds the attribute labels
        for the model
        """
        cdef crfsuite_dictionary_t *attrs
        cdef CRFDict attr_dict = CRFDict()
        if self.model.get_attrs(self.model, &attrs):
            raise ValueError("Could not create dictionary")
        attr_dict.set_internal_dict(attrs)
        attrs.release(attrs)
        return attr_dict

    def get_tagging_data_from_file(self, filename):
        """Get tagging data from an IWA file based on the model.

        The difference between loading a dataset in this way and just loading
        the filename directly is that when loading tagging data, only the
        attributes which are in the training set are loaded.  In addition,
        the attributes within the training data and tagging data are
        guaranteed to be associated  with the same integer ids.

        Parameters
        ----------
        filename: string
            location of IWA file from which data will be read

        Returns
        -------
        data : CRFDataset object
            the data appropriate for tagging.
        """
        cdef CRFDataset data = CRFDataset()

        data.add_groups_from_files_for_tagging(filename,
                                               self.get_attr_dict(),
                                               self.get_label_dict(),
                                               logfile=None)
        return data

    def get_tagger(self):
        """
        Create a CRFTagger object based on the model.
        """
        return CRFTagger(self)

    def dump(self, fpo=None):
        """dump the contents of the model in human-readable format

        Parameters
        ----------
        fpo: stream or filename, default = sys.stdout
            out stream in which to print the results
        """
        if fpo is None:
            fpo = sys.stdout
        elif type(fpo) == file:
            pass
        else:
            fpo = open(fpo, 'w')

        self.model.dump(self.model, PyFile_AsFile(fpo))


cdef class CRFTagger(object):
    """Wrapper for the crfsuite tagger structure

    Parameters
    ----------
    model: CRFModel object
        The training model from which to build the tagger structure
    """

    cdef crfsuite_tagger_t *tagger
    cdef CRFModel model

    def __init__(self, CRFModel model):
        self.model = model
        if model.model.get_tagger(model.model, &self.tagger):
            raise ValueError("could not get tagger from model")

    def __cinit__(self):
        self.tagger = NULL

    def __dealloc__(self):
        if self.tagger != NULL:
            self.tagger.release(self.tagger)
            self.tagger = NULL

    def get_label_dict(self):
        return self.model.get_label_dict()

    def get_attr_dict(self):
        return self.model.get_attr_dict()

    def get_tagging_data_from_file(self, filename):
        return self.model.get_tagging_data_from_file(filename)

    #TODO: create a "get_tagging_data_from_array" routine here
    #      and in CRFModel.

    #TODO : add keywords to tag which allow the full range of
    #       crfsuite tag options
    #
    def tag(self, CRFDataset data, evaluate=True, quiet=False):
        """Tag the data based on trained model

        Parameters
        ----------
        data: CRFDataset object
            The test data.  Note that the attribute dictionary of test data
            should be identical to the attribute dictionary of the training
            data.
            For data from file, this can be assured by reading the
            file using CRFTagger.get_tagging_data_from_file(filename)
            For data from an array, this can be assured by initializing
            the array with the correct attributes.

        evaluate: boolean, default=False
            If true, report the performance of the model on the data

        quiet : boolean, default=False
            If true, suppress output

        Returns
        -------
        TODO : store other results (probability, marginal probabilities, etc)
        output : ndarray, shape=(data.n_samples,)
            The label ids for the fit associated with each item.
        """
        cdef crfsuite_data_t *data_ptr = &data.data
        cdef crfsuite_instance_t inst
        cdef crfsuite_item_t item
        cdef crfsuite_attribute_t cont
        cdef crfsuite_evaluation_t eval

        cdef floatval_t score
        cdef int i
        cdef int N = 0
        cdef int L = data.n_labels

        cdef np.ndarray output = np.zeros(data.n_samples, dtype=int)
        cdef int *output_arr = <int*> output.data
        cdef int output_idx = 0

        crfsuite_evaluation_init(&eval, L)

        fpo = sys.stdout

        for i from 0 <= i < data_ptr.num_instances:
            inst = data_ptr.instances[i]

            # Set the instance to the tagger.
            if self.tagger.set(self.tagger, &inst):
                raise ValueError("could not set instance to tagger")

            # Obtain the viterbi label sequence.
            if self.tagger.viterbi(self.tagger,
                                   output_arr + output_idx, &score):
                raise ValueError("could not generate label sequence")

            N += 1

            # Accumulate the tagging performance.
            if evaluate:
                crfsuite_evaluation_accmulate(&eval, inst.labels,
                                              output_arr + output_idx,
                                               inst.num_items)

            if not quiet:
                output_tagging_result(PyFile_AsFile(fpo), self.tagger,
                                      &inst, output_arr + output_idx,
                                      data_ptr.labels, score)

            output_idx += inst.num_items

        crfsuite_evaluation_finish(&eval)
        return output


######################################################################
# Interface routines

def crfsuite_learn(CRFDataset crf_data,
                   fittype="1d",
                   algorithm="lbfgs",
                   param_dict={},
                   model_file='',
                   split=0,
                   holdout=-1,
                   cross_validate=False,
                   log_to_file=False,
                   logbase="log.crfsuite",
                   rseed=None):
    """Perform Conditional Random Field learning

    This calls python wrappers of the library crfsuite

    Parameters
    ----------
    crf_data : CRFData object
        training data

    fittype: string
        specify a graphical model (default = '1d')

        - '1d' : first-order Markov CRF with state and transition features;
          transition features are not conditioned on observations

    algorithm: string
        specify a training algorithm (default = 'lbfgs')

        - 'lbfgs' : L-BFGS with L1/L2 regularization
        - 'l2sgd' : SGD with L2-regularization
        - 'ap' :    Averaged Perceptron
        - 'pa' :    Passive Aggressive
        - arow' :   Adaptive Regularization of Weights (AROW)

    param_dict: dictionary
        specify parameters specific to each algorithm

    model_file: string
        specify the filename to which the model will be saved.
        if not specified, the model will be saved to a temporary location.

    split: integer
        For split = N > 0, split the instances into N groups. This option
        is useful for holdout evaluation and cross validation. (default = 0)

    holdout: integer
        For holdout = M >= 0 , use the M-th data group (zero-indexed)
        for holdout evaluation and the rest for training.
        If split is specified, then these are the N data groups.
        If split is not specified, the input files are the N data groups.
        (default = -1)

    cross_validate: boolean
        for split = N, repeat holdout evaluations for #i in {1, ..., N} groups:
        this is N-fold cross validation.
        If split is specified, then these are the N data groups.
        If split is not specified, the input files are the N data groups.
        (default = False)

    log_to_file: boolean
        write the training log to a file instead of to STDOUT;
        The filename is determined automatically by the training
        algorithm, parameters, and source files (default=False)

    logbase: string
        set the base name for a log file (used if ``log_to_file = True``)
        default = 'log.crfsuite'

    rseed: integer
        if specified, use this random seed for shuffling instances

    Returns
    -------

    Examples
    --------

    Notes
    -----
    """
    trainer = CRFTrainer(fittype, algorithm, **param_dict)

    cdef int n, i, ret

    #----------------------------------------------------------------------
    # Open a logfile if necessary
    fpo = sys.stdout

    if log_to_file:
        fname = '%s_%s' % (logbase, algorithm)
        for opt in sorted(param_dict.keys()):
            fname += '_%s=%s' % (opt, param_dict[opt])

        fpo = open(fname, "w")

    #----------------------------------------------------------------------
    # Log the start time
    fpo.write("Start time of the training: %s\n\n" % time.asctime())

    #----------------------------------------------------------------------
    # Split data if necessary
    if split > 0:
        crf_data.split(split)

    #----------------------------------------------------------------------
    # Report the statistics of the training data.
    crf_data.dump(fpo)

    #----------------------------------------------------------------------
    # Start training
    model = trainer.train(crf_data, cross_validate, holdout, model_file, fpo)

    #----------------------------------------------------------------------
    # Log the end time
    fpo.write("End time of the training: %s\n\n" % time.asctime())

    if model_file:
        fpo.write("Saved model to %s\n" % model_file)

    return model
