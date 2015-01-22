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

# Seismic local linear regression function setup: takes a INITIAL(station, phase)
# and predicts the time residual using the SIZE nearest neighbors on EXAMPLES,
# and using the LOOKUP dictionary for computational efficiency. Returns the
# appropriate klocal_linear_regression_lookup function, whose one argument
# is a location. This is used as a hypothesis function.
def residual_regression_setup(initial, size, examples, lookup):
    return lambda x : klocal_linear_regression_lookup(initial[0], initial[1], 
												(float(x[6]), float(x[7])), 
												examples, size, lookup)

#  Returns the error between a residual prediction FX and the actual residual at X.
def residual_err(x, fx):
	return util.one_norm(float(x[10]), fx[0]) 

# Returns the argument X of the seismic metric as a hashable key.
def residual_hash(x):
	comp1 = tuple([tuple([str(x[0][0][0]), str(x[0][0][1])]), x[0][1], x[0][2]])
	comp2 = tuple([tuple([str(x[1][1]), str(x[1][2])]), x[1][9], x[1][5]])
	return comp1, comp2

# This function returns all the nearest neighbors we will ever need on the residual
# problem, that is, given INITIAL(station, phase) pair, writes a dictionary of nearest
# neighbors in the EXAMPLES set to seismicnn.csv and is returned. If LOOKUP is true, the 
# function simply returns the dictionary read from seismicnn.csv.
def seismic_nn(initial, examples, lookup):
	if lookup:
		reader = csv.reader(open('seismicnn.csv', 'rb'))
		return dict(x for x in reader)
	writer = csv.writer(open('seismicnn.csv', 'wb'))
	nn = defaultdict(lambda: [])
	count, length = 0, np.floor(len(examples)/float(100))
	point_clusters = defaultdict(lambda: [])
	print 'computing point clusters on ' + str(len(examples)) + ' points.'
	for point in range(len(examples)):
		point_clusters[(examples[point][5], examples[point][9])] = point_clusters[(examples[point][5], examples[point][9])] + [point]
	print 'computed seismic point clusters, of which there are ' + str(len(point_clusters.items()))
	passcount = 0
	for point in examples:
		if count % length == 0:
			print "computed " + str(passcount) + '% of examples'
			passcount += 1
		count += 1
		loc = (float(point[1]), float(point[2]))
		station, phase = initial
		metric = memoized(seismic_metric, residual_hash)
		if nn[(loc, phase, station)] == []:
			nn[(loc, phase, station)] = nn_list((loc, phase, station), point_clusters[(station, phase)], metric)
			key, value = (loc, phase, station), nn[(loc, phase, station)]
			writer.writerow([str(key), value])
	return nn

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
	nn = k_nearest_neighbors(k, (x, phase, station), data, seismic_metric, hashfn)
	nn = [float(event[10]) for event in nn]
	return np.mean(nn), np.var(nn)

# Analogous to the above function, but does nearest neighbors lookup instead of computing.
def klocal_linear_regression_lookup(station, phase, x, data, k, lookup):
	nn = lookup[(x, phase, station)][:k]
	residuals = [float(event[10]) for event in nn]
	return np.mean(residuals), np.var(residuals)

def localLinearRegressionForP1(x, data):
   pass

def localLinearRegressionForP2(x, data):
   pass

def localLinearRegressionForS1(x, data):
   pass

def localLinearRegressionForS2(x, data):
   pass


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
