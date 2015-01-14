import pylab
from matplotlib import pyplot
import numpy
import ProbabilityModel

# Plots 10,000 values from a Univariate Normal r.v. with mu, sigma as the respective
# mean and variance.
def plot_normal(mu, sigma):
	y = ProbabilityModel.UnivariateNormal(mu, sigma)
	x = []
	for y1 in range(10000):
		x = x + [y.sample()]
	pyplot.hist(x)
	pylab.show()

# Plots 10,000 values from a multidimensional Normal r.v. with Mu, Sigma matrices of
# mean and variance.
def plot_mdnormal(mu, sigma):
	z = ProbabilityModel.MultiVariateNormal(mu, sigma)
	x, y = [], []
	for y1 in range(10000):
		samp = z.sample()
		x = x + [samp[0][0]]
		y = y + [samp[1][0]]
	pyplot.hist2d(x, y, bins=(100, 100))
	pylab.show()

# Plots 10,000 values from a categorical distribution with atomic probabilites vector
# m
def plot_categorical(m):
	size = tuple([i for i in range(len(m)+1)])
	n = ProbabilityModel.Categorical(m)
	x = [n.sample() for i in range(10000)]
	pyplot.hist(x, bins=size, normed=True)
	pylab.show()


# Plots 10,000 values from a mixture model with centers (+-1, +-1) and covariance the 
# identity and prints the probability that a sample from the distribution will lie in the 
# unit circle with center (.1, .2).
def plot_mixture():
	centers = [numpy.array([[1], [1]]), numpy.array([[-1], [1]]),
            numpy.array([[1], [-1]]), numpy.array([[-1], [-1]])]
	z = [ProbabilityModel.MultiVariateNormal(centers[i], numpy.array([[1, 0], [0, 1]])) for i in range(len(centers))]
	ap = [.25, .25, .25, .25]
	x, y = [], []
	q = ProbabilityModel.MixtureModel(ap, z)
	yes = 0
	for y1 in range(10000):
		samp = q.sample()
		x0, y0 = samp[0][0], samp[1][0] 
		x = x + [x0]
		y = y + [y0]
		if numpy.sqrt(pow(x0 - .1, 2) + pow(y0 - .2, 2)) < 1:
			yes += 1
	print float(yes)/10000	
	pyplot.hist2d(x, y, bins=(50, 50))
	pylab.show()

plot_mixture()
