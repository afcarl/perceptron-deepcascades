# Deep Cascades with Perceptron

Some relevant files:

* *perceptron.py*: The perceptron and kernel perceptron algorithms, written by Mathieu Blondel, October 2010
* *pdc.py*: Train and test deep cascades with perceptron
* *compare_dumps.py*: Compare best classifiers from certain dumps and choose the best one from them
* *pdc_multiprocess.py*: Use pdc.py with compare_dumps.py to run as multiple processes, bypassing the GIL
* *split_sets.py*: Split a dataset into training and testing sets
* *generate_folds.py*: Split a training set into three folds of training and testing sets
* *crossvalidate.py*: Train on the folds per a particular gamma and dataset and dump the result
* *evaluate.py*: Evaluate the dumps from crossvalidate.py to determine the crossvalidation results
* *svm_test.py*: Cross-validate for degree and C for SVM and get testing error
