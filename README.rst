==========
pyCRFsuite
==========

This is a python wrapper for crfsuite, a fast implementation of Conditional
Random Fields

Authors
-------

- Jake Vanderplas <vanderplas@astro.washington.edu>


Installation
------------

Currently the package is set-up only for in-place installation.  It requires
the ``crfsuite`` library to be installed: see
http://www.chokkan.org/software/crfsuite/

Once this is installed, simply type ``make`` in the head directory.

Testing
-------
There are a few basic test scripts in the head directory.  ``test.py`` will
read a small dataset from ``example_files``, then run a basic training and
tagging operation.  ``crfsuite_test.sh`` runs the same operation using the
command-line frontend provided by crfsuite.  To compare the results of the
training and tagging, run ``compare_output.sh``.  This will print all the
places where the tagging results differ.

TODO
----
This is still a very incomplete wrapper.  Search ``TODO`` within
``src/crfsuite.pyx`` to see some issues that need to be addressed.

Issues
------
There are a few 'features' in crfsuite that make efficient python wrapping
difficult.

- **Model File Output**: as currently written, crfsuite writes the result of
  a training directly to a binary file.  The library is not configured to
  allow writing the model to memory.  This means that a python wrapper must
  write the model to disk, then read the model into memory before performing
  any tagging operation.  It would be better if the model could be saved
  directly to a CRFsuite model structure, though when dealing with the very
  large datasets for which crfsuite is designed, it's clear why the author
  made the choice he did.

- **Memory mapping**: as currently written, crfsuite data is not stored in
  contiguous arrays.  This means that there is no way to map a crfsuite data
  structure to a numpy array, and any input to crfsuite will need to be
  copied in memory.  Addressing this would require significant upstream
  changes: the ``crfsuite_item_t`` structure would have to use an array of
  floats and an array of ints rather than an array of attribute structures.