#This module contains code for class VariableRanker, which uses a random forest to
# estimate the importance of each variable with respect to a particular index of the
# data. There is a generator for a variable ranker, as well as a range function.
import random_forest
import csv
from numpy.random import uniform
import math

# Returns a v(ariable)r(anker) and a set of labels using a VAR_VECTOR of variables,
# a PREDICTION_INDEX in the data with a set of DIST_CLASSES, since this is classification.
# the DATA_FILE contains the comma seperated values, and the default trees are set to 500.
def vr_generator(var_vector, prediction_index, dist_classes, data_file, trees=500):
  Data = [events for events in csv.reader(open(data_file))]
  labels = Data[0]
  rest = Data[2:]
  vr = VariableRanker(rest, var_vector, prediction_index, dist_classes, trees)
  return vr, labels

# Returns the range of a continuous or discrete VAR(int, {'c', 'd'}) in EXAMPLES
def variable_range(examples, var):
  if var[1] == 'd':
    range = set()
    for datum in examples:
      range.add(datum[var[0]])
    return range
  else:
    range_min, range_max = 0, 0
    for datum in examples:
      data_val = float(datum[var[0]])
      range_min, range_max = min(range_min, data_val), max(range_max, data_val)
    return (range_min, range_max)

# A VariableRanker is a random forest together with a variable ranking function,
# which is used to estimate the relative importance of a set of variables in a 
# set of data with respect to a prediction index and some distinguished class
class VariableRanker:
  forest = []
  # Creates a ranker.
  def __init__(self, data, variables, prediction_index, dist_classes, trees=500):
    self.trees = trees
    self.data = data
    self.variables = variables
    self.prediction_index = prediction_index
    self.dist_classes = dist_classes
  # This constructs a random forest from which predictions are drawn in the ranking function
  def grow_trees(self, regrow=False):
    if self.forest == []:
      mtry = int(math.floor(math.sqrt(len(self.variables))))
      self.forest = random_forest.RandomForest(self.data, self.trees, mtry, self.variables, self.prediction_index, random_forest.rf_attr_fn, self.dist_classes, len(self.data))
      print self.trees, '  have been grown using a set of ', len(self.variables), ' variables.'
    elif regrow:
      mtry = int(math.floor(math.sqrt(len(self.variables))))
      self.forest = random_forest.RandomForest(self.data, self.trees, mtry, self.variables, self.prediction_index, random_forest.rf_attr_fn, self.dist_classes, len(self.data))
      print self.trees, ' have been regrown using a set of ', len(self.variables), ' variables.' 
    else:
      print "Already a forest in place, add regrow=True to override."
  # Ranks the variables in this variable ranker.
  def variable_ranking(self):
    self.grow_trees()
    dist_classes = self.dist_classes
    oob = self.forest.oob_set_generator()
    oob_length, First, elt_vals, var_vals = len(oob), True, {}, {}
    succ_rate, dist_succ_rate, dist_order =  0, 0, 0
    for var in self.variables:
      var_range = list(variable_range(self.data, var))
      range_len = len(var_range)
      print var
      permution = None
      permuted_succ, perm_dist_succ =  0, 0
      for elts in oob:
        if First:
          actual = self.data[elts][self.prediction_index]
          elt_vals[elts] = actual
          predicted = self.forest.test_predict(self.data[elts], elts)
          if actual in dist_classes:
            dist_order += 1
          if actual == predicted:
            succ_rate += 1
            if actual in dist_classes:
              dist_succ_rate += 1
        if var[1] == 'd':
          permution = int(math.floor(uniform(0, 1)*range_len))
          permution = var_range[permution]
        else:
          permution = uniform(0, 1)*(var_range[1] - var_range[0])
        perm_tuple = self.data[elts][:var[0]] + [permution] + self.data[elts][var[0]+1:]
        permuted_prediction = self.forest.predict(perm_tuple)
        actual = elt_vals[elts]
        if actual == permuted_prediction:
          permuted_succ += 1
          if actual in dist_classes:
            perm_dist_succ += 1
      if First:
        succ_rate = float(succ_rate)/oob_length
        dist_succ_rate = float(dist_succ_rate)/dist_order
      First = False
      permuted_succ = float(permuted_succ)/oob_length
      perm_dist_succ = float(perm_dist_succ)/dist_order
      print "Originally a ", succ_rate, " success rate, with permution to ", permuted_succ
      print "A difference of ", succ_rate - permuted_succ
      print "WRT Distinguised classes, a success rate of:", dist_succ_rate, 'with permution to ', perm_dist_succ
      print "A difference of ", dist_succ_rate - perm_dist_succ
      var_vals[var] = succ_rate - permuted_succ
      var_vals[(var, 'd')] = dist_succ_rate - perm_dist_succ
    for x in var_vals.items():
      print x[0], x[1]
