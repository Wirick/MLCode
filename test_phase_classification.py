import unittest
import csv
import util
import numpy as np

def test_histogram():
	assert util.discrete_histogram('toy.csv', ['a']) == [('0', 4, .4), ('6', 6, .6)]

def test_confusion():
	x = util.confusion_matrix('toy.csv', 'a', 'b')
	assert np.array_equal(x[0], np.matrix([[0.75, 0.25], [0, 1.0]])) == True
	assert x[1] == ['0', '6']

def test_confusion2():
	x = util.discrete_histogram('trainingData2.csv', ['phase'])
	print x
	assert 1 == 2
