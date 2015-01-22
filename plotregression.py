import matplotlib.pyplot as plt
import util
import random
import evalWrapper
import csv
import regression

DATA = [events for events in csv.reader(open('trainingData.csv'))]

# Trying to get a sense of how this k-fold validation function is working.
# this generates points (x, fx) where f = x^2, then it does cross validation
# on local linear regression to try and find the appropriate value of k.
def plot_plane_regression_validation():
	data = []
	for i in range(1000):
		x = 100*random.uniform(0, 1)
		fx = pow(x, 2)
		data = data + [(x, fx)]
	wrap = evalWrapper.EvalWrapper([], util.plane_regression_setup)
	x, y = util.k_fold_cross_validation(5, wrap, data, util.plane_err)
	plt.plot([i for i in range(len(x))], x, 'ro')
	plt.plot([i for i in range(len(y))], y, 'bo')
	plt.show()

def plot_seismic_regression():
	data = DATA[1:]
	wrap = evalWrapper.EvalWrapper(['908', 'P'], regression.residual_regression_setup, data, regression.seismic_nn)
	x, y = util.k_fold_cross_validation(10, wrap, data, regression.residual_err)
	plt.plot([i for i in range(len(x))], x, 'ro')
	plt.plot([i for i in range(len(y))], y, 'bo')
	plt.show() 

if __name__ == '__main__':
	plot_seismic_regression()
