# Utility functions to compute ml problems.
import numpy
import heapq
import random
import ProbabilityModel

# We want some memoization here
class memoized(object):
	def __init__(self, function, arghash):
		self.function = function
		self.arghash = arghash
		self.reference = {}
	def __call__(self, *args):
		hsh = self.arghash(args)
		if hsh in self.reference:
			return self.reference[hsh]
		else:
			value = self.function(*args)
			self.reference[hsh] = value
			return value


# k nearest neighbors: Given an integer K, a query point X, a 
# set of DATA(csv.reader), and a DISTANCE function which computes the distance 
# from a point of data to x. Computes the k nearest neighbors to x via the distance function.
def k_nearest_neighbors(k, x, data, distance, arghash):
	distance = memoized(distance, arghash)
	data = [events for events in data]
	nneighbors = []
	for event in data:
		dist = distance(x, event)
		if dist == numpy.inf:
			continue
		if len(nneighbors) < k:
			heapq.heappush(nneighbors, (-dist, event))
		else:
			heapq.heappushpop(nneighbors, (-dist, event))
	return [event[1] for event in nneighbors]

# k-fold cross-validation: Given an integer K, a LEARNER(EvalWrapper), and EXAMPLES, 
# performs k-fold cross validation. errFn gives the difference between a prediction
# and a data point
def k_fold_cross_validation(k, learner, examples, errFn):
	eT, eV = [], []
	for size in range(1, 10):
		eT0, eV0 = cross_validation(k, size, learner, examples, errFn) 
		eT.append(eT0)
		eV.append(eV0)
		print eT, eV
	return eT, eV 
		
# Performs k fold cross validation with dimension SIZE  with a given 
# LEARNER on a set of EXAMPLES, returns the mean of the sample error
# for a given hypothesis calculated with the ERRFN.
def cross_validation(k, size, learner, examples, errfn):
	fold_errT, fold_errV = 0, 0
	partitions = partition(examples, k, True)
	for i in range(k):
		train, val = partitions.next()
		hypothesis = learner.get_hypothesis(size, train)
		for x in train:
			fold_errT += errfn(x, hypothesis(x))
		for x in val:
			fold_errV += errfn(x, hypothesis(x))
	return fold_errT/len(train), fold_errV/len(val)

# Returns a generator which produces the K-fold partitions of the LST of 
# training examples. Randomizes examples before partitioning when RAND is
# True, and leaves in order otherwise.
def partition(lst, k, rand):
	if rand:
		random.shuffle(lst)
	slices = [lst[i::k] for i in range(k)]
	for i in range(k):
		val = slices[i]
		train = []
		for j in range(k):
			if j == i:
				continue
			train = train + slices[j]
		yield train, val

# Plane local regression function which takes a real X and predicts f(x) using
# the K nearest neighbors on EXAMPLES and the 2-norm as the distance function
def plane_local_regression(x, k, examples):
	nn = k_nearest_neighbors(k, x, examples, plane_regression_norm)
	nn = [item[1] for item in nn]
	return numpy.mean(nn), numpy.var(nn)

# Returns a function which serves as a hypothesis for a plane local linear regression
# model. Initial is presumed empty in this case.
def plane_regression_setup(initial, size, examples):
	return lambda x : plane_local_regression(x[0], size, examples)

# Given a data point X in R^2 and the result of a plane local regression 
# prediction FX, computes the 1 norm of FX and X[1], which is the actual value.
def plane_err(x, fx):
	return one_norm(x[1], fx[0])

# The 1 norm of x and y elements of the real numbers. 
def one_norm(x, y):
	return abs(x - y)

# The 2-d euclidean norm of X0 and X1.
def two_norm(x0, x1):
	return numpy.sqrt(pow(x0[0] - x1[0], 2) + pow(x0[1] - x1[1], 2))

# This norm measures the distance between a real X0 and X1 
# in R^2 by taking the 1-norm of x0 and x1[0]
def plane_regression_norm(x0, x1):
	return one_norm(x0, x1[0])
