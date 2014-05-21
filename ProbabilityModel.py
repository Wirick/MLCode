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
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):

    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma

    # Returns a sample from a single variable normal distribution
    def sample(self):
        x = super(UnivariateNormal, self).sample()
        mu, sigma, pi, exp = self.mu, self.sigma, math.pi, math.exp
        return 1/(sigma * math.pow(2*pi,1/2)) * exp(-math.pow(x-mu,2)/2*math.pow(sigma,2))

    
    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.mu = matrix(Mu)
        self.sigma = matrix(Sigma)
        self.D = Mu.shape[0]

    # Returns a sample from a multivariate Gaussian distribution.
    def sample(self):
        rands = [[super(MultiVariateNormal, self).sample()] for x in range(self.D)]
        mu, sigma, pi, exp, D = self.mu, self.sigma, math.pi, math.exp, self.D
        covFactor = linalg.det(linalg.matrix_power(sigma,1/2))
        dimFactor = math.pow(2*pi,D/2)
        conj = (rands-mu).T * sigma.I * (rands-mu)
        return 1/(covFactor*dimFactor) * exp( (-.5)* conj )
        
    

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

    def sample(self):
        rand, size = super(Categorical, self).sample(), self.size
        dx = 1/(float(size))
        for x in range(size):
            if (rand-(x+1)*dx) < 0:
                return self.ap[x]
        return nil


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
        self.size = len(ap)

    def sample(self):
        ap, pm, size = self.ap, self.pm, self.size
        return sum([ap[x]*pm[x].sample() for x in range(size)])
