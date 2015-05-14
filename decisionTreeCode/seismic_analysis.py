import decision_tree
from collections import defaultdict
from operator import itemgetter
import csv
import math

Data = [events for events in csv.reader(open('trainingData.csv'))][2:1000]
def ordered_importance(attributes, examples):
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
tree = decision_tree.DecisionTreeLearner(Data[:500], [(58, 'd'), (59, 'c'), (39, 'c'), (48, 'c'), (49, 'c'), (50, 'c')], [], 28, ordered_importance)
for i in range(100):
 print Data[501 + i][28], tree.decide_element(Data[501 + i])

