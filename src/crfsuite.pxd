from libc.stdio cimport FILE

cdef extern from "stdarg.h":
    ctypedef struct va_list:
        pass
    cdef int vfprintf(FILE* stream, char* format, va_list arg)

cdef extern from "crfsuite.h":
    ctypedef double floatval_t

    #------------------------------------------------------------
    # Object interfaces and utilities
    #------------------------------------------------------------

    # CRFsuite model interface
    cdef struct tag_crfsuite_model
    ctypedef tag_crfsuite_model crfsuite_model_t
    
    # CRFsuite trainer interface
    cdef struct tag_crfsuite_trainer
    ctypedef tag_crfsuite_trainer crfsuite_trainer_t
    
    # CRFsuite tagger interface
    cdef struct tag_crfsuite_tagger
    ctypedef tag_crfsuite_tagger crfsuite_tagger_t
    
    # CRFsuite dictionary interface
    cdef struct tag_crfsuite_dictionary
    ctypedef tag_crfsuite_dictionary crfsuite_dictionary_t
    
    # CRFsuite parameter interface
    cdef struct tag_crfsuite_params
    ctypedef tag_crfsuite_params crfsuite_params_t

    #------------------------------------------------------------
    # Dataset (attribute, item, instance, dataset)
    #------------------------------------------------------------
    ctypedef struct crfsuite_attribute_t:
        int aid
        floatval_t value

    ctypedef struct crfsuite_item_t:
        int num_contents
        int cap_contents
        crfsuite_attribute_t *contents

    ctypedef struct crfsuite_instance_t:
        int num_items
        int cap_items
        crfsuite_item_t *items
        int *labels
        int group

    ctypedef struct crfsuite_data_t:
        int num_instances
        int cap_instances
        crfsuite_instance_t *instances
        crfsuite_dictionary_t *attrs
        crfsuite_dictionary_t *labels

    #------------------------------------------------------------
    # Evaluation utilities
    #------------------------------------------------------------
    ctypedef struct crfsuite_label_evaluation_t:
        int num_correct
        int num_observation
        int num_model
        floatval_t precision
        floatval_t recall
        floatval_t fmeasure
    
    ctypedef struct crfsuite_evaluation_t:
        int num_labels
        crfsuite_label_evaluation_t* tbl
        
        int item_total_correct
        int item_total_num
        int item_total_observation
        int item_total_model
        floatval_t item_accuracy

        int inst_total_correct
        int inst_total_num
        floatval_t inst_accuracy

        floatval_t macro_precision
        floatval_t macro_recall
        floatval_t macro_fmeasure

    #------------------------------------------------------------
    # callback function
    #------------------------------------------------------------
    ctypedef int (*crfsuite_logging_callback)(void *user,
                                              char *format,
                                              va_list args)

    #------------------------------------------------------------
    # Structure definitions
    #------------------------------------------------------------
    cdef struct tag_crfsuite_model:
        void *internal
        int nref
        int (*addref)(crfsuite_model_t* model)
        int (*release)(crfsuite_model_t* model)
        int (*get_tagger)(crfsuite_model_t* model,
                          crfsuite_tagger_t** ptr_tagger)
        int (*get_labels)(crfsuite_model_t* model,
                          crfsuite_dictionary_t** ptr_labels)
        int (*get_attrs)(crfsuite_model_t* model,
                         crfsuite_dictionary_t** ptr_attrs)
        int (*dump)(crfsuite_model_t* model,
                    FILE* fpo)

    cdef struct tag_crfsuite_trainer:
        void *internal
        int nref
        int (*addref)(crfsuite_trainer_t* trainer)
        int (*release)(crfsuite_trainer_t* trainer)
        crfsuite_params_t* (*params)(crfsuite_trainer_t* trainer)
        void (*set_message_callback)(crfsuite_trainer_t* trainer,
                                     void* user,
                                     crfsuite_logging_callback cbm)
        int (*train)(crfsuite_trainer_t* trainer,
                     crfsuite_data_t* data,
                     char* filename,
                     int holdout)


    cdef struct tag_crfsuite_tagger:
        void* internal
        int nref
        int (*addref)(crfsuite_tagger_t* tagger)
        int (*release)(crfsuite_tagger_t* tagger)
        int (*set)(crfsuite_tagger_t* tagger,
                   crfsuite_instance_t *inst)
        int (*length)(crfsuite_tagger_t* tagger)
        int (*viterbi)(crfsuite_tagger_t* tagger,
                       int *labels,
                       floatval_t *ptr_score)
        int (*score)(crfsuite_tagger_t* tagger,
                     int *path,
                     floatval_t *ptr_score)
        int (*lognorm)(crfsuite_tagger_t* tagger,
                       floatval_t *ptr_norm)
        int (*marginal_point)(crfsuite_tagger_t *tagger,
                              int l,
                              int t,
                              floatval_t *ptr_prob)
        int (*marginal_path)(crfsuite_tagger_t *tagger,
                             int *path,
                             int begin,
                             int end,
                             floatval_t *ptr_prob)


    cdef struct tag_crfsuite_dictionary:
        void* internal
        int nref
        int (*addref)(crfsuite_dictionary_t* dic)
        int (*release)(crfsuite_dictionary_t* dic)
        int (*get)(crfsuite_dictionary_t* dic,
                   char *str)
        int (*to_id)(crfsuite_dictionary_t* dic,
                     char *str)
        int (*to_string)(crfsuite_dictionary_t* dic,
                         int id,
                         char **pstr)
        int (*num)(crfsuite_dictionary_t* dic)
        void (*free)(crfsuite_dictionary_t* dic,
                     char *str)

    cdef struct tag_crfsuite_params:
        void* internal
        int nref
        int (*addref)(crfsuite_params_t* params)
        int (*release)(crfsuite_params_t* params)
        int (*num)(crfsuite_params_t* params)
        int (*name)(crfsuite_params_t* params,
                    int i,
                    char **ptr_name)
        int (*set)(crfsuite_params_t* params,
                   char *name,
                   char *value)
        int (*get)(crfsuite_params_t* params,
                   char *name,
                   char **ptr_value)
        int (*set_int)(crfsuite_params_t* params,
                       char *name,
                       int value)
        int (*set_float)(crfsuite_params_t* params,
                         char *name,
                         floatval_t value)
        int (*set_string)(crfsuite_params_t* params,
                          char *name,
                          char *value)
        int (*get_int)(crfsuite_params_t* params,
                       char *name,
                       int *ptr_value)
        int (*get_float)(crfsuite_params_t* params,
                         char *name,
                         floatval_t *ptr_value)
        int (*get_string)(crfsuite_params_t* params,
                          char *name,
                          char **ptr_value)
        int (*help)(crfsuite_params_t* params,
                    char *name,
                    char **ptr_type,
                    char **ptr_help)
        void (*free)(crfsuite_params_t* params,
                     char *str)

    #------------------------------------------------------------
    # Function definitions
    #------------------------------------------------------------

    #-----------------------
    # Creation of structures
    #-----------------------

    # instance allocation
    cdef int crfsuite_create_instance(char *iid,
                                      void **ptr)

    # instance creation from file
    cdef int crfsuite_create_instance_from_file(char *filename,
                                                void **ptr)

    # tagging instance creation from model file
    cdef int crfsuite_create_tagger(char *filename,
                                    crfsuite_tagger_t** ptr_tagger,
                                    crfsuite_dictionary_t** ptr_attrs,
                                    crfsuite_dictionary_t** ptr_labels)

    #---------------------
    # Attribute structures
    #---------------------

    # initialize attribute structure
    cdef void crfsuite_attribute_init(crfsuite_attribute_t* attr)

    # set an attribute and its value
    cdef void crfsuite_attribute_set(crfsuite_attribute_t* attr,
                                     int aid,
                                     floatval_t value)

    # copy the content of an attribute structure
    cdef void crfsuite_attribute_copy(crfsuite_attribute_t* dst,
                                      crfsuite_attribute_t* src)

    # swap the contents of two attribute structures
    cdef void crfsuite_attribute_swap(crfsuite_attribute_t* x,
                                      crfsuite_attribute_t* y)

    #---------------------
    # Item structures
    #---------------------

    # initialize an item structure
    cdef void crfsuite_item_init(crfsuite_item_t* item)

    # initialize an item structure with the number of attributes
    cdef void crfsuite_item_init_n(crfsuite_item_t* item,
                                   int num_attributes)

    # uninitialize an item structure
    cdef void crfsuite_item_finish(crfsuite_item_t* item)

    # copy the contents of an item structure
    cdef void crfsuite_item_copy(crfsuite_item_t* dst,
                                 crfsuite_item_t* src)

    # swap the contents of two item structures
    cdef void crfsuite_item_swap(crfsuite_item_t* x,
                                 crfsuite_item_t* y)

    # append an attribute to the item structure
    cdef int  crfsuite_item_append_attribute(crfsuite_item_t* item,
                                             crfsuite_attribute_t* attr)

    # check whether the item has no attribute
    cdef int crfsuite_item_empty(crfsuite_item_t* item)

    #---------------------
    # Instance structures
    #---------------------

    # initialize an instance structure
    cdef void crfsuite_instance_init(crfsuite_instance_t* seq)

    # uninitialize an instance structure
    cdef void crfsuite_instance_finish(crfsuite_instance_t* seq)

    # copy the content of an instance structure
    cdef void crfsuite_instance_copy(crfsuite_instance_t* dst,
                                     crfsuite_instance_t* src)

    # swap the contents of two instance structures
    cdef void crfsuite_instance_swap(crfsuite_instance_t* x,
                                     crfsuite_instance_t* y)

    # append an (item, label) pair to the instance structure
    cdef int crfsuite_instance_append(crfsuite_instance_t* seq,
                                      crfsuite_item_t* item,
                                      int label)

    # check whether the instance has no item
    cdef int crfsuite_instance_empty(crfsuite_instance_t* seq)

    #---------------------
    # Dataset structures
    #---------------------

    # initialize a dataset structure
    cdef void crfsuite_data_init(crfsuite_data_t* data)

    # initialize a dataset structure with the number of instances
    cdef void crfsuite_data_init_n(crfsuite_data_t* data,
                                   int n)

    # uninitialize a dataset structure
    cdef void crfsuite_data_finish(crfsuite_data_t* data)

    # copy the content of a dataset structure
    cdef void crfsuite_data_copy(crfsuite_data_t* dst,
                                 crfsuite_data_t* src)

    # swap the contents of two dataset structures
    cdef void crfsuite_data_swap(crfsuite_data_t* x,
                                 crfsuite_data_t* y)

    # append an instance to the dataset structure
    cdef int crfsuite_data_append(crfsuite_data_t* data,
                                  crfsuite_instance_t* inst)

    # obtain the maximum length of the instances in the dataset
    cdef int crfsuite_data_maxlength(crfsuite_data_t* data)

    # obtain the total number of items in the dataset
    cdef int crfsuite_data_totalitems(crfsuite_data_t* data)

    #----------------------
    # Evaluation structures
    #----------------------

    # initialize an evaluation structure.
    cdef void crfsuite_evaluation_init(crfsuite_evaluation_t* eval,
                                       int n)

    # uninitialize an evaluation structure.
    cdef void crfsuite_evaluation_finish(crfsuite_evaluation_t* eval)

    # reset an evaluation structure.
    cdef void crfsuite_evaluation_clear(crfsuite_evaluation_t* eval)

    # accmulate the correctness of the predicted label sequence
    cdef int crfsuite_evaluation_accmulate(crfsuite_evaluation_t* eval,
                                           int* reference,
                                           int* prediction,
                                           int T)

    # finalize the evaluation result.
    cdef void crfsuite_evaluation_finalize(crfsuite_evaluation_t* eval)


    # print the evaluation result.
    cdef void crfsuite_evaluation_output(crfsuite_evaluation_t* eval,
                                         crfsuite_dictionary_t* labels,
                                         crfsuite_logging_callback cbm,
                                         void *user)

    #------------------------------------------------------------
    # Miscellaneous definitions and functions
    #------------------------------------------------------------

    # increments the value of the integer variable as an atomic operation
    cdef int crfsuite_interlocked_increment(int *count)

    # decrements the value of the integer variable as an atomic operation
    cdef int crfsuite_interlocked_decrement(int *count)


cdef extern from "Python.h":
    FILE* PyFile_AsFile(object)


#cdef extern from "readdata.h":
#    cdef int read_data(FILE *fpi, FILE *fpo,
#                       crfsuite_data_t* data, int group)


# defined in iwa.c
ctypedef struct iwa_string_t:
    size_t size
    size_t offset
    char* value

cdef extern from "iwa.h":
    cdef struct tag_iwa_token:
        int type
        char* attr
        char* value
    ctypedef tag_iwa_token iwa_token_t

    cdef struct tag_iwa:
        FILE* fp
        iwa_token_t token
        char* buffer
        char* offset
        char* end
        iwa_string_t attr
        iwa_string_t value
    ctypedef tag_iwa iwa_t

    cdef enum iwa_enumeration:
        IWA_BOI
        IWA_EOI
        IWA_ITEM
        IWA_NONE
        IWA_EOF
    
    cdef iwa_t* iwa_reader(FILE* fp)

    cdef iwa_token_t* iwa_read(iwa_t* iwa)

    cdef void iwa_delete(iwa_t* iwa)
        
