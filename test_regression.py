import unittest
import evalWrapper
from regression1 import *
import ProbabilityModel
from util import *
import numpy
import csv

DATA = [event for event in csv.reader(open('trainingData.csv'))]
# Tests the example in the regression.py klocallinearregression function
def test_klocal1():
	print "Testing klocalLinearRegression"
	print "args: 908, P, [10.25, -15.50], DATA, 6"
	print klocalLinearRegression('908', 'P', [10.25, -15.50], DATA, 6)
	assert True == False

# Tests the seismic metric in the util package, since the phase of the data
# is the same as the query point, we should expect to see a finite distance.
def test_seismic1():
	assert seismic_metric(((0, 0), 'P', '908'), DATA[1]) < numpy.inf

## If the phase of the data is not the same as the phase of the data point we 
# expect for the seismic metric to give a distance of infinity.
def test_seismic2():
	assert seismic_metric(((0, 0), 'S', '908'), DATA[1]) == numpy.inf

# If the station of the data is not the same as the phase of the data
# we expect a metric of infinity.
def test_seismic3():
	assert seismic_metric(((0, 0), 'P', '909'), DATA[1]) == numpy.inf

# Tests the k nearest neighbors function.
def test_knn1():
	assert set(k_nearest_neighbors(3, 1, [0, 1, 2, 3, 4, 5, 6], one_norm)) == set([1, 0, 2])
	assert set(k_nearest_neighbors(4, 1, [0, 1, 2, 3, 4, 5, 6], one_norm)) == set([0, 1, 2, 3])

def test_knn1():
	hashfn = residual_hash
	data = DATA
	station, phase, x, k = '1069', 'P', (64.771446, -146.88665), 1
	nn = k_nearest_neighbors(k, (x, phase, station), data, seismic_metric, hashfn)
	assert True == True

# Tests the parition function.
def test_partition1():
	x = partition([1, 2, 3, 4, 5], 2, False) 
	assert x.next() == ([2, 4], [1, 3, 5])
	assert x.next() == ([1, 3, 5], [2, 4])
	y = partition([1, 2, 3, 4, 5], 3, False)
	assert y.next() == ([2, 5, 3], [1, 4])
	assert y.next() == ([1, 4, 3], [2, 5])

# Tests the 2-norm function.
def test_2norm():
	x0 = (1, 1)
	x1 = (4, 5)
	assert two_norm(x0, x1) == 5

# Tests the cross validation function
def test_cross_validation():
	data = []
	for i in range(1000):
		x = 4*random.uniform(0, 1)
		fx = pow(x, 2)
		data = data + [(x, fx)]
	wrap = evalWrapper.EvalWrapper([], plane_regression_setup)
	#k_fold_cross_validation(5, wrap, data, plane_err)
	assert True == True

