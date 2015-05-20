# NOTE: Skeleton provided for the UC Berkeley Machine learning class in Fall 2011,
# which can be found at http://www.cs.berkeley.edu/~russell/classes/cs194/f11/assignments/a0/sampler.py

from scipy import stats
from numpy import random, matrix, linalg
import math

class ProbabilityModel(object):

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        return random.uniform()

# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2).
class UnivariateNormal(ProbabilityModel):

    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma

    # Returns a sample from a single variable normal distribution
	# using the Box Muller transformation
    def sample(self):
		x, y = random.uniform(), random.uniform()
		z = math.sqrt(-2*math.log(x))*math.cos(2*math.pi*y)
		return self.sigma*z + self.mu

# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability
# measure is defined by the density function
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):

    # Initializes a multivariate normal probability model object
    # parameterized by Mu (numpy.array of size D x 1) expectation vector
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.Mu = Mu
        self.Sigma = Sigma
        self.D = [UnivariateNormal(0, 1) for x in range(Mu.shape[0])]

    # Returns a sample from a multivariate Gaussian distribution. It is a theorem
    # that a vector of D independent normally distributed random variables has
    # a natural inclusion into the space of multidimensional normal variables.
    # i.e given Mu, and a vector Z of D ind. normal variables then there exists
    # a matrix A such that Mu + AZ ~ N(Mu, Sigma) for some unique N. We use the
    # Cholesky decomposition to compute this matrix
    def sample(self):
		A = linalg.cholesky(self.Sigma)
		Z = [[x.sample()] for x in self.D]
		Mu = self.Mu
		return Mu + A.dot(Z)

# The sample space of this probability model is the finite discrete set {0..k-1}, and
# the probability measure is defined by the atomic probabilities
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete)
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
		self.ap = ap
		self.size = len(ap)
		self.cutoffs = []
		current_cut = 0
		for i in range(self.size):
			self.cutoffs = self.cutoffs + [ap[i] + current_cut]
			current_cut += ap[i]

	# So here the idea is just to draw a uniformly distributed number and
	# use the cuttoffs to decide when to stop.
    def sample(self):
		x = random.uniform()
		for i in range(self.size):
			if x > self.cutoffs[i]:
				continue
			return i

# The sample space of this probability model is the union of the sample spaces of
# the underlying probability models, and the probability measure is defined by
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):

    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of
    # probability models pm
    def __init__(self,ap,pm):
        self.ap = ap
        self.pm = pm
        self.model = Categorical(ap)
	
	# There is a bijection between the subspace of mixture models mod their atomic
	# probability vectors and the space of categorical models given by the mapping
	# [Mixture(ap)] |--> Categorical(ap). first we sample the categorical model, and
	# use the result as the index of the probability model we will use for this sample.
    def sample(self):
		which = self.model.sample()
		samp = self.pm[which].sample()
		return samp


