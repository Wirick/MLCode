import random_forest
from numpy.random import uniform
import math

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

class VariableRanker:
  trees = 500
  forest = []
  def __init__(self, data, variables, prediction_index, dist_classes):
    self.data = data
    self.variables = variables
    self.prediction_index = prediction_index
    self.dist_classes = dist_classes
  def grow_trees(self, regrow=False):
    if self.forest == []:
      mtry = int(math.floor(math.sqrt(len(self.variables))))
      self.forest = random_forest.RandomForest(self.data, self.trees, mtry, self.variables, self.prediction_index, random_forest.rf_attr_fn, self.dist_classes, len(self.data))
      print self.trees, ' have been grown using a set of ', len(self.variables), ' variables.'
    elif regrow:
      mtry = int(math.floor(math.sqrt(len(self.variables))))
      self.forest = random_forest.RandomForest(self.data, self.trees, mtry, self.variables, self.prediction_index, random_forest.rf_attr_fn, self.dist_classes, len(self.data))
      print self.trees, ' have been regrown using a set of ', len(self.variables), ' variables.' 
    else:
      print "Already a forest in place, add regrow=True to override."
  def variable_ranking(self):
    self.grow_trees()
    oob = self.forest.oob_set_generator()
    for var in self.variables:
      var_range = list(variable_range(self.data, var))
      print var, ' range is ', var_range
      permution = None
      succ_rate, permuted_succ = 0, 0
      if var[1] == 'd':
        permution = int(math.floor(uniform(0, 1)*len(var_range)))
        permution = var_range[permution]
      else:
        permution = uniform(0, 1)*(var_range[1] - var_range[0])
      for elts in oob:
        actual = self.data[elts][self.prediction_index]
        predicted = self.forest.test_predict(self.data[elts], elts)
        perm_tuple = self.data[elts][:var[0]] + [permution] + self.data[elts][var[0]+1:]
        permuted_prediction = self.forest.test_predict(perm_tuple, elts)
        if actual == predicted:
          succ_rate += 1
        if actual == permuted_prediction:
          permuted_succ += 1
      succ_rate, permuted_succ = float(succ_rate)/len(oob), float(permuted_succ)/len(oob)
      print "Originally a ", succ_rate, " success rate, with permution to ", permuted_succ
