all: crfsuite.so

crfsuite.so: src/crfsuite.pyx
	python setup.py build_ext --inplace