#This module contains code for class VariableRanker, which uses a random forest to
# estimate the importance of each variable with respect to a particular index of the
# data. There is a generator for a variable ranker, as well as a range function.
import random_forest
import csv
from numpy.random import uniform
import math

def vr_generator(var_vector, prediction_index, dist_classes, data_file, trees=500, attr_fn=random_forest.rf_attr_fn, imp=random_forest.rf_gini_split):
    """ Returns a v(ariable)r(anker) and a set of labels using a VAR_VECTOR of variables,
            a PREDICTION_INDEX in the data with a set of DIST_CLASSES, since this is classification.
            the DATA_FILE contains the comma seperated values, and the default trees are set to 500. """
    Data = [events for events in csv.reader(open(data_file))]
    labels = Data[0]
    rest = Data[2:]
    vr = VariableRanker(rest, var_vector, prediction_index, dist_classes, trees, attr_fn, imp)
    return vr, labels

def variable_range(examples, var):
    """ Returns the range of a continuous or discrete VAR(int, {'c', 'd'}) in EXAMPLES. """
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
# set of data with respect to a prediction index and some distinguished class.
class VariableRanker:
    forest = []
    def __init__(self, data, variables, prediction_index, dist_classes, trees, attr_fn, imp):
        """ Creates a ranker. """
        self.attr_fn = attr_fn
        self.trees = trees
        self.data = data
        self.variables = variables
        self.prediction_index = prediction_index
        self.dist_classes = dist_classes
        self.importance_fn = imp

    def grow_trees(self, regrow=False):
        """ This constructs a random forest from which predictions are drawn in the ranking function. """
        if self.forest == [] or regrow:
            mtry = int(math.floor(math.sqrt(len(self.variables))))
            data, trees, var, pred_index = self.data, self.trees, self.variables, self.prediction_index
            attr_fn, dist_classes, order, imp = self.attr_fn, self.dist_classes, len(self.data), self.importance_fn
            self.forest = random_forest.RandomForest(data, trees, mtry, var, pred_index, attr_fn, dist_classes, order, imp)
            print self.trees, '  have been grown using a set of ', len(self.variables), ' variables.'
        else:
            print "Already a forest in place, add regrow=True to override."

    def variable_ranking(self):
        """ Ranks the variables in this variable ranker. """
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
        var_vals = sorted(var_vals.items(), key=lambda x: x[1], reverse=True)
        for x in var_vals:
            print x[0], x[1]
