from random import shuffle, sample
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import svm
from numpy import asarray, delete
import pickle

def train_and_validation(training_set, val_size=10000):
  print "Training and Validation Creation"
  ind = range(len(training_set[0]))
  inds = sample(ind, val_size)
  training_labels, training_data = asarray(training_set[0][inds]), training_set[1][inds]
  inds = delete(ind, inds)
  validation_labels = asarray(training_set[0][inds])
  validation_data = training_set[1][inds]
  return [(training_labels, training_data), (validation_labels, validation_data)]

training_orders = [100, 200, 500, 1000, 2000, 5000, 10000]

def default_orders():
  return training_orders

def get_linear_svm(training_set, regularization=1.0):
  svc = svm.LinearSVC(C=regularization)
  labels, data = training_set
  svc.fit(data, labels.ravel())
  return svc

def linear_svm_classify(training_set, validation_set, training_order=training_orders, save_file='linear_svm.pkl'):
  svc = svm.LinearSVC()
  training_labels, training_points = training_set
  validation_labels, validation_points = validation_set
  print "Training Classifiers!"
  scores, train_magnitude = [], range(len(training_labels))
  for order in training_order:
    inds = sample(train_magnitude, order)
    svc.fit(training_points[inds], training_labels[inds].ravel())
    scores.append(svc.score(validation_points, validation_labels.ravel()))
    name = str(order) + "_" + save_file
    pickle.dump(svc, open(name, 'wb'))
  pickle.dump(validation_labels, open('val_labels_' + save_file, 'wb'))
  pickle.dump(validation_points, open('val_points_' + save_file, 'wb'))
  return scores, training_order

def bullet_plot(data, labels, title, xval, yval, save_location):
  plt.clf()
  plt.plot(data, labels, 'ro')
  plt.title(title)
  plt.xlabel(xval)
  plt.ylabel(yval)
  plt.savefig(save_location)

def plot_confusion(matrix, title, xtick, ytick, save_file):
  plt.clf()
  plt.matshow(matrix)
  plt.title(title)
  plt.colorbar()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.xticks(xtick)
  plt.yticks(ytick)
  plt.savefig(save_file)

# Returns the confusion matrix of ATTR1 and ATTR2 and drawing from DATA, 
# also returns respectively the list of row and column values that index
# the matrix.
def confusion_matrix(data, attr1, attr2):
	row_vals = discrete_histogram(data, [attr1])
	row_keys = key_function(row_vals)
	col_keys = key_function(discrete_histogram(data, [attr2]))
	hist2 = discrete_histogram(data, [attr1, attr2], lambda x: tuple(x))
	rows = defaultdict(lambda: [])
	for key in row_vals:
		rows[key[0]] = [0 for i in range(len(col_keys))]
	for key in hist2:
		row, col = key[0]
		row_freq = row_vals[row_keys.index(row)][1]
		rows[row][col_keys.index(col)] = round(float(key[1])/row_freq, 4)
	matr = []
	for key in row_keys:
		matr.append(rows[key])
	return np.asarray(matr), row_keys, col_keys

# Returns a list of (value, abs_frequency, relative_freq) tuples using
# attributes drawn from ATTR(list) combined using
# some COMBINE function and drawing from DATA(csv). The function takes
# a data point and returns the value of the attribute applicable to that
# datum. The default combine function is assuming one attribute is being examined.
def discrete_histogram(data, attr, combine=lambda x: x[0]):
	order = len(data)
	attr_index = attr
	histogram = defaultdict(lambda: 0)
	tuples = []
	for point in data:
		if point == []:
			continue
		attr_vals = combine([point[x] for x in attr_index])
		histogram[attr_vals] += 1
	for key in histogram:
		tuples = tuples + [(key, histogram[key], histogram[key]/float(order))]
	return tuples

# Given a HISTOGRAM, returns a list of the values x[0] of the points x in 
# the histogram.
def key_function(histogram):
	keys = []
	for point in histogram:
		keys = keys + [point[0]]
	return keys

# K-fold cross-validation: Given an integer K, a LEARNER(EvalWrapper), and EXAMPLES, 
# performs k-fold cross validation. ERRFN gives the difference between a prediction
# and a data point
def k_fold_cross_validation(k, learner, examples, errFn, sizes):
	eT, eV = [], []
	for size in sizes:
		eT0, eV0 = cross_validation(k, size, learner, examples, errFn) 
		eT.append(eT0)
		eV.append(eV0)
		print eT, eV, str(size) + '-size'
	return eT, eV 
		
# Performs K fold cross validation with dimension SIZE with a given 
# LEARNER on a set of EXAMPLES, returns the mean of the sample error
# for a given hypothesis calculated with the ERRFN.
def cross_validation(k, size, learner, examples, errfn):
	fold_errT, fold_errV = 0, 0
	partitions = partition(examples, k, True)
	for i in range(k):
		percent, kfold_errT, kfold_errV = 0, 0, 0
		print str(size) + '-size, fold ' + str(i+1)
		train, val = partitions.next()
		hypothesis = learner.get_hypothesis(size, train)
		comp = np.floor(float(len(train[0]))/10)
		for i in range(len(train[0])):
			if i % comp == 0:
				percent += 10 
			kfold_errT += errfn(train[0][i], hypothesis.predict(train[1][i]))
		for x in range(len(val[0])):
			kfold_errV += errfn(val[0][x], hypothesis.predict(val[1][x]))
		train_err, val_err = float(kfold_errT)/len(train[0]), float(kfold_errV)/len(val[0])
		fold_errT, fold_errV = fold_errT + train_err, fold_errV + val_err
		print 'Errors: Validation = ' + str(val_err) + ' Training = ' + str(train_err)
	return fold_errT/k, fold_errV/k

# Returns a generator which produces the K-fold partitions of the LST of 
# training examples. Randomizes examples before partitioning when RAND is
# True, and leaves in order otherwise.
def partition(examples, k, rand):
	inds = range(len(examples[0]))
	if rand:
		shuffle(inds)
	slices = [inds[i::k] for i in range(k)]
	labels, data = examples
	for i in range(k):
		val = slices[i]
		train = []
		for j in range(k):
			if j == i:
				continue
			train = train + slices[j]
		train_lab, train_dat = np.asarray([labels[x] for x in train]), [data[x] for x in train]
		val_lab, val_dat = np.asarray([labels[x] for x in val]), [data[x] for x in val]
		yield [(train_lab, train_dat), (val_lab, val_dat)]
