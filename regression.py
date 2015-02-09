from numpy import *
import sys
import util
from scipy import stats
from numpy import matrix
from numpy import linalg
from collections import defaultdict
import numpy as np
import csv

csv.field_size_limit(sys.maxsize)
DATA = [events for events in csv.reader(open('trainingData.csv'))]
k_nearest_neighbors = util.k_nearest_neighbors
nn_list = util.nn_list
memoized = util.memoized

## Calculates the great circle distance between two point on the earth's
## surface in degrees. loc1 and loc2 are pairs of latitude and longitude . E.g.
## dist((10,0), (20, 0)) gives 10
def dist(loc1, loc2):
   DEG2RAD = np.pi / 180
   RAD2DEG = 180 / np.pi
   lat1, lon1  = loc1
   lat2, lon2  = loc2
   return np.arccos(np.sin(lat1 * DEG2RAD) * np.sin(lat2 * DEG2RAD)
                    + np.cos(lat1 * DEG2RAD) * np.cos(lat2 * DEG2RAD)
                    * np.cos((lon2 - lon1) * DEG2RAD)) * RAD2DEG

# Describes the distance function for the seismic wave detection
# problem. If DATA(seismic event entry) doesn't have the same phase or station
# as X(location, phase, station), the function returns infinity, else it returns the great
# circle distance between data's event location and x's location
def seismic_metric(x, data):
	d, distance = dist, np.inf
	loc, phase, station = x
	if phase == data[9] and station == data[5]:
		x0 = (float(data[1]), float(data[2]))
		distance = d(loc, x0)
	return distance

# This metric is for the purposes of training, where you are manipulating
# pieces of your dataset instead of just a loc, phase, station situation.
def eval_seismic_metric(x, data):
	print (x[1], x[2]), x[9], x[5], data
	return seismic_metric(((float(x[1]), float(x[2])), x[9], x[5]), data)

# If POINT has the same phase and station as IV(station, phase), returns 
# True, returns False otherwise.
def seismic_shard(point, iv):
	station, phase = iv
	if point[9] == phase and point[5] == station:
		return True
	return False

# Seismic local linear regression function setup: takes a INITIAL(station, phase)
# and predicts the time residual using the SIZE nearest neighbors on EXAMPLES,
# and using the LOOKUP dictionary for computational efficiency. Returns the
# appropriate klocal_linear_regression_lookup function, whose one argument
# is a location. This is used as a hypothesis function.
def residual_regression_setup(initial, size, examples, lookup):
	if len(lookup) == 0:
		return lambda x : klocalLinearRegression(initial[0], initial[1], 
								(float(x[6]), float(x[7])), examples, size)
	else:
		return lambda x : k_local_regression_lookup((float(x[6]), float(x[7])), lookup)

#  Returns the error between a residual prediction FX and the actual residual at X.
def residual_err(x, fx):
	return util.one_norm(float(x[10]), fx[0]) 

# Returns the argument X of the seismic metric as a hashable key.
def residual_hash(x):
	comp1 = tuple([tuple([str(x[0][0][0]), str(x[0][0][1])]), x[0][1], x[0][2]])
	comp2 = tuple([tuple([str(x[1][1]), str(x[1][2])]), x[1][9], x[1][5]])
	return comp1, comp2

# local linear regression with lookup, assuming X is in LOOKUP, returns the mean
# of the residuals in lookup[x].
def k_local_regression_lookup(x, lookup):
	return np.mean([float(y[10]) for y in lookup[x]]) 

## Estimates the residual time of a query point x(lat, lon) using local
## linear regression.
## Inputs: station(str), phase(str), x(list of 2 floats:lat, lon), data(list)
## E.g. localLinearRegression('908','P', [10.25,-15.50] , 6)
## Outputs estimate(float) and varestimate(float) using the return command, e.g.
## return estimate, varestimate
# The idea is to run the nearest neighbors function to find the nearest events for which
# the right phase was observed at the designated station. Then average the resulting residuals
def klocalLinearRegression(station, phase, x, data, k):
	hashfn = residual_hash
	nn = k_nearest_neighbors(k, (x, phase, station), data, seismic_metric)
	nn = [float(event[10]) for event in nn]
	return np.mean(nn), np.var(nn)

def localLinearRegressionForP1(x, data):
   pass

def localLinearRegressionForP2(x, data):
   pass

def localLinearRegressionForS1(x, data):
   pass

def localLinearRegressionForS2(x, data):
   pass

# So now we are going to want to predict the variance of
# our predictions using the variance formula in the assignment.
#

## Estimate the residual time using locally weighted
## regression with Gaussian or Laplacian kernel
## Outputs estimate(float)
def localWeightedRegression(station, phase, x, data):
   pass

# Finds the top two stations in both S and P phase detections and
# returns a dictionary whose keys are 'S', 'P' and whose values
# are sets of two stations with the top detections, along with their
# detection counts. Takes DATA, a cvs.reader object for events.
def findTopStations(data):
	counts = defaultdict(lambda: 0)
	for event in data:
		if event[9] == 'P':
			counts[(event[5], 'P')] += 1
		elif event[9] == 'S':
			counts[(event[5], 'S')] += 1
	top = defaultdict(lambda: [])
	for key in counts:
		phasevals = top[key[1]]
		if len(phasevals) == 0:
			phasevals = [(key[0], counts[key])]
		elif len(phasevals) == 1:
			if phasevals[0][1] < counts[key]:
				phasevals = [(key[0], counts[key])] + [phasevals[0]]
		else:
			insert = (key[0], counts[key])
			for i in range(len(phasevals)):
				if phasevals[i][1] < insert[1]:
					phasevals[i], insert = insert, phasevals[i]
		top[key[1]] = phasevals
	return top

# print findTopStations(DATA)
#print 'Running regression. Station 1069, phase P::: ' + str(klocalLinearRegression('1069', 'P', [0, 0], DATA, 6))
#print 'running 908, P::: ' + str(klocalLinearRegression('908', 'P', [0, 0], DATA, 6))
#print '1069, S::: ' + str(klocalLinearRegression('1069', 'S', [0, 0], DATA, 6))
#print '908, S::: ' + str(klocalLinearRegression('908', 'S', [0, 0], DATA, 6))
