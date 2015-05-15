import decision_tree
import cPickle
import rf_variable_ranker
import random_forest
from collections import defaultdict
from operator import itemgetter
import csv
import math

Data = [events for events in csv.reader(open('trainingData.csv'))][2:]
def ordered_importance(attributes, examples, predict_index, positive_classes):
  attr = attributes[0]
  if attr[1] == 'd':
    vals = defaultdict(lambda: 0)
    for e in examples:
      vals[e[attr[0]]] += 1
    return attr, [keys for keys in vals]
  if attr[1] == 'c':
    ordered = sorted(examples, key=lambda x: float(x[attr[0]]))
    length = int(math.ceil(len(ordered)/float(2)))
    vals = [(ordered[0][attr[0]], ordered[length][attr[0]]), (ordered[length][attr[0]], ordered[len(ordered) - 1][attr[0]])]
    return attr, vals
def normal_attr(attr, elt):
  lst = list(attr)
  lst.remove(elt)
  return lst
var = rf_variable_ranker.VariableRanker(Data[:500], [(58, 'd'), (59, 'c'), (39, 'c'), (48, 'c'), (49, 'c'), (50, 'c')], 28, ['P', 'Pg', 'Pn', 'PKP'])
var.grow_trees()
var.variable_ranking()
#trees = random_forest.RandomForest(Data, 100, 3, [(58, 'd'), (59, 'c'), (39, 'c'), (48, 'c'), (49, 'c'), (50, 'c')], 28, random_forest.rf_attr_fn, ['P', 'Pg', 'Pn', 'PKP'], 500)
#correct = 0
#print trees.oob_set_generator()
#for i in range(500):
# actual, predict = Data[5001 + i][28], trees.test_predict(Data[5001 + i], 5001+i)
# if actual == predict:
#   correct += 1
# print actual, predict
#print correct/float(500)
