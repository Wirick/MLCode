# Utility functions to compute ml problems.
import numpy
import heapq
import random
import ProbabilityModel
from collections import defaultdict

# We want some memoization here: This class is a 
# simple memoization of a function.
class memoized(object):

    def __init__(self, function, arghash):
        """ Constructor of a memoization of a FUNCTION via 
            an ARGHASH which makes the keys nice. """
        self.function = function
        self.arghash = arghash
        self.reference = {}

    def __call__(self, *args):
        """ Calling the function first hashes the optional
            ARGS, and then if the computation is stored
            in self.reference, returns that value, otherwise
            stores and returns the function call with args. """
        print args
        hsh = self.arghash(args)
        if hsh in self.reference: return self.reference[hsh]
        else:
            print self.function
            value = self.function(*args)
            self.reference[hsh] = value
            return value

def get_component(data, shard, iv):
    """ So we want a way of splitting data by certain attributes, so DATA is a list of 
        data. Points in data are fed into the SHARD function with the IV to determine
        if the point has the desired attributes. """
    distinguished = []
    for point in data:
        belongs = shard(point, iv)
        if belongs: distinguished = distinguished + [point]
    return distinguished

def k_nearest_neighbors(k, x, data, distance):
    """ K nearest neighbors: Given an integer K, a query point X, a 
        set of DATA(csv.reader), and a DISTANCE function which computes the distance 
        from a point of data to x, returns the k nearest neighbors to x via the distance function. """
    data = [events for events in data]
    nneighbors = []
    for event in data:
        dist = distance(x, event)
        if dist == numpy.inf: continue
        if len(nneighbors) < k: heapq.heappush(nneighbors, (-dist, event))
        else: heapq.heappushpop(nneighbors, (-dist, event))
    nneighbors = sorted(nneighbors, key=lambda event: -event[0])
    return [event[1] for event in nneighbors]

def k_nearest_neighbors_lookup(k, data, metric):
    """ K nearest neighbors lookup: Returns a dictionary whose keys are the
        elements of DATA and whose values are the K nearest neighbors via the 
        METRIC. """
    nn = defaultdict(list)
    for element in data:
        neighbors = k_nearest_neighbors(k+1, element, data, metric)
        nn[element] = str(neighbors[1:])
    return nn

def nn_reduce(size, lst):
    """ Function which returns lst[:size], it's a reduction function for lookup
        computation. """
    return lst[:size]

def nn_list(x, examples, distance, length):
    """ K nearest neighbors list: So if you keep recomputing the k nearest neighbors
        you are going to be waiting a long time for cross validation. This returns a 
        sorted list of LENGTH nearest neighbors of a point X in examples with finite
        distance. """
    nn = k_nearest_neighbors(length, x, examples, distance)
    return nn

def k_fold_cross_validation(k, learner, examples, errFn):
    """ K-fold cross-validation: Given an integer K, a LEARNER(EvalWrapper), and EXAMPLES, 
        performs k-fold cross validation. ERRFN gives the difference between a prediction
        and a data point. """
    eT, eV = [], []
    for size in range(1, 31):
        eT0, eV0 = cross_validation(k, size, learner, examples, errFn) 
        eT.append(eT0)
        eV.append(eV0)
        print eT, eV, str(size) + '-nn'
    return eT, eV 
            
def cross_validation(k, size, learner, examples, errfn):
    """ Performs K fold cross validation with dimension SIZE with a given 
        LEARNER on a set of EXAMPLES, returns the mean of the sample error
        for a given hypothesis calculated with the ERRFN. """
    fold_errT, fold_errV = 0, 0
    partitions = partition(examples, k, False)
    for i in range(k):
        percent, kfold_errT, kfold_errV = 0, 0, 0
        print str(size) + '-nn, fold ' + str(i)
        train, val = partitions.next()
        hypothesis = learner.get_hypothesis(size, train)
        print 'training on set of ' + str(len(train)) + ' values'
        comp = numpy.floor(float(len(train))/10)
        for i in range(len(train)):
            if i % comp == 0: percent += 10 
            kfold_errT += errfn(train[i], hypothesis(train[i]))
        print 'validating on set of ' + str(len(val)) + ' values'
        for x in val: kfold_errV += errfn(x, hypothesis(x))
        train_err, val_err = kfold_errT/len(train), kfold_errV/len(val)
        fold_errT, fold_errV = fold_errT + train_err, fold_errV + val_err
        print 'Errors: Validation = ' + str(val_err) + ' Training = ' + str(train_err)
    return fold_errT/k, fold_errV/k

def partition(lst, k, rand):
    """ Returns a generator which produces the K-fold partitions of the LST of 
        training examples. Randomizes examples before partitioning when RAND is
        True, and leaves in order otherwise. """
    if rand: random.shuffle(lst)
    slices = [lst[i::k] for i in range(k)]
    for i in range(k):
        val = slices[i]
        train = []
        for j in range(k):
            if j == i: continue
            train = train + slices[j]
        yield train, val

def record_eval(train, val, filename):
    """ Just a method for storing evaluation data so that things don't need to 
        be recomputed. """
    f = open(filename, 'w')
    f.write('trainerr\n')
    for x in train: f.write(str(x) + '\n')
    f.write('valerr\n')
    for x in val: f.write(str(x) + '\n')
    f.close()

def plane_local_regression(x, k, examples):
    """ Plane local regression function which takes a real X and predicts f(x) using
        the K nearest neighbors on EXAMPLES and the 1-norm as the distance function. """
    nn = k_nearest_neighbors(k, x, examples, plane_regression_norm)
    nn = [item[1] for item in nn]
    return numpy.mean(nn), numpy.var(nn)

def plane_regression_setup(initial, size, examples):
    """ Returns a function which serves as a hypothesis for a plane local linear regression
        model. Initial is presumed empty in this case."""
    return lambda x : plane_local_regression(x[0], size, examples)

def plane_err(x, fx):
    """ Given a data point X in R^2 and the result of a plane local regression 
        prediction FX, computes the 1 norm of FX and X[1], which is the actual value. """
    return one_norm(x[1], fx[0])

def one_norm(x, y):
    """ The 1 norm of x and y elements of the real numbers. """
    return abs(x - y)

def two_norm(x0, x1):
    """ The 2-d euclidean norm of X0 and X1."""
    return numpy.sqrt(pow(x0[0] - x1[0], 2) + pow(x0[1] - x1[1], 2))

def plane_regression_norm(x0, x1):
    """ This norm measures the distance between a real X0 and X1 
        in R^2 by taking the 1-norm of x0 and x1[0]."""
    return one_norm(x0, x1[0])

def discrete_histogram(data, attr, combine=lambda x: x[0]):
    """ Returns a list of (value, abs_frequency, relative_freq) tuples using
        attributes drawn from ATTR(list) combined using some COMBINE function
        and drawing from DATA(csv). The function takes a data point and returns
        the value of the attribute applicable to that datum. The default combine
        function is assuming one attribute is being examined. """
    data = [events for events in csv.reader(open(data))]
    attributes, data = data[0], data[1:]
    order = len(data)
    attr_index = [attributes.index(x) for x in attr]
    histogram = defaultdict(lambda: 0)
    tuples = []
    for point in data:
        if point == []: continue
        attr_vals = combine([point[x] for x in attr_index])
        histogram[attr_vals] += 1
    for key in histogram:
        tuples = tuples + [(key, histogram[key], histogram[key]/float(order))]
    return tuples

def complete_histogram(data):
    """ Returns a dictionary whose keys are the attributes in csv on the first
        line of DATA, the values are default dictionary whose keys are attribute values
        and whose values are the absolute frequency of the value in the dataset. """
    data = [events for events in csv.reader(open(data))]
    histogram = defaultdict(lambda: defaultdict(lambda: 0))
    keys = data[0]
    data = data[1:]
    for point in data:
        for i in range(len(keys)):
            histogram[keys[i]][point[i]] += 1
    print histogram
        
def confusion_matrix(data, attr1, attr2):
    """ Returns the confusion matrix of ATTR1 and ATTR2 and drawing from DATA, 
        also returns respectively the list of row and column values that index
        the matrix. """
    row_vals = discrete_histogram(data, [attr1])
    row_keys = key_function(row_vals)
    col_keys = key_function(discrete_histogram(data, [attr2]))
    hist2 = discrete_histogram(data, [attr1, attr2], lambda x: tuple(x))
    rows = defaultdict(lambda: [])
    for key in row_vals: rows[key[0]] = [0 for i in range(len(col_keys))]
    for key in hist2:
        row, col = key[0]
        row_freq = row_vals[row_keys.index(row)][1]
        rows[row][col_keys.index(col)] = float(key[1])/row_freq
    matr = []
    for key in row_keys: matr.append(rows[key])
    return np.matrix(matr), row_keys, col_keys

def key_function(histogram):
    """ Given a HISTOGRAM, returns a list of the values x[0] of the points x in 
        the histogram."""
    keys = []
    for point in histogram: keys = keys + [point[0]]
    return keys
