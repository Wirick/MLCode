from numpy.random import uniform
from collections import defaultdict
import operator
import math
import decision_tree

def random_forest_generator(examples, attributes, parent_examples, predict_index, importance, attr_fn):
  pass

def p_random_attributes(attributes, p):
  lst = list(attributes)
  for i in range(int(len(attributes) - p)):
    pick = math.floor(uniform(0, 1)*len(attributes))
    if pick in lst:
      lst.remove(pick)
    else:
      i -= 1
  return tuple(lst)

def discrete_gini(attribute, examples, predict_index, positive_classes):
  order_vals, counts = defaultdict(lambda: 0), defaultdict(lambda: 0)
  order = len(examples)
  for i in range(order):
    if examples[i][predict_index] in positive_classes:
      order_vals[examples[i][attribute]] += 1
    counts[examples[i][attribute]] += 1
  val_order = len(order_vals.items())
  dist_order, dist_set = int(math.ceil(math.sqrt(val_order))), []
  vals = defaultdict(lambda: 0)
  for key, value in order_vals.items():
    vals[key] = value / float(counts[key])
  sorted_vals = sorted(vals.items(), key=operator.itemgetter(1), reverse=True) 
  side1count, side1pos, side2count, side2pos = 0, 0, 0, 0
  yes_split = sorted_vals[:dist_order]
  no_split = sorted_vals[dist_order:]
  for item in vals.items():
    if item in yes_split:
      side1count += counts[item[0]]
      side1pos += order_vals[item[0]]
    else:
      side2count += counts[item[0]]
      side2pos += order_vals[item[0]]
  gini_score = pow(2, side1pos/float(side1count + 1)) + pow(float(side1count - side1pos)/(side1count + 1), 2)
  gini_score += pow(2, side2pos/float(side2count + 1)) + pow(float(side2count - side2pos)/(side2count + 1), 2)
  if len(yes_split) == 1:
    if len(no_split) == 1:
      return gini_score, (yes_split[0][0], no_split[0][0])
    return gini_score, tuple(yes_split[0][0])
  return gini_score, (tuple([x[0] for x in yes_split]), tuple([x[0] for x in no_split]))
  

def continuous_gini(attribute, examples, predict_index, positive_classes):
  ordered = sorted(examples, key=operator.itemgetter(attribute))
  side2len, side1len, side2pos, side1pos, order = len(ordered), 0, 0, 0, len(ordered)
  gini_max, gini_max_index, last_checked = 0, 0, None
  for x in ordered:
    if x[predict_index] in positive_classes:
      side2pos += 1
  for i in range(1, order):
    if ordered[i][predict_index] in positive_classes:
      side1pos += 1
      side2pos -= 1
    side1len += 1
    side2len -= 1
    if ordered[i][predict_index] in positive_classes and last_checked:
      continue
    elif ordered[i][predict_index] not in positive_classes and not last_checked:
      continue
    elif last_checked == None:
      last_checked = ordered[i][predict_index] in positive_classes
    else:
      new_gini = pow(2, side1pos/float(side1len)) + pow(float(side1len - side1pos)/side1len, 2)
      new_gini += pow(2, side2pos/float(side2len)) + pow(float(side2len - side2pos)/side2len, 2) 
      if new_gini > gini_max:
        gini_max = new_gini
        gini_max_index = i 
  split_val = (float(ordered[gini_max_index][attribute]) + float(ordered[gini_max_index - 1][attribute]))/float(2)
  val1 = (float(ordered[0][attribute]), split_val)
  val2 = (split_val, float(ordered[order - 1][attribute]))
  return gini_max, (val1, val2)

def rf_gini_split(attributes, examples, predict_index, positive_classes):
  attr = p_random_attributes(attributes, math.ceil(math.sqrt(len(attributes))))
  gini_scores = defaultdict(lambda: 0)
  for attribute in attr:
    if attribute[1] == 'c':
      gini_scores[attribute] = continuous_gini(attribute[0], examples, predict_index, positive_classes)
    else:
      gini_scores[attribute] = discrete_gini(attribute[0], examples, predict_index, positive_classes)
  split = max([x for x in gini_scores.items()], key=operator.itemgetter(1))
  return split[0], split[1][1]

def rf_attr_fn(attributes, elt):
  return attributes

class RandomForest:
  forest = []
  def __init__(self, examples, trees, mtry, attributes, predict_index, attr_fn, dist_classes, order):
    for i in range(trees):
      picks = set()
      for j in range(int(math.floor(order*1/float(3)))):
        pick = math.floor(uniform(0, 1)*order)
        picks.add(pick)
      picks = list(picks)
      elements = [examples[int(x)] for x in picks]
      tree = decision_tree.dt_generator(elements, attributes, [], predict_index, rf_gini_split, rf_attr_fn, dist_classes)
      tree.set_growth_indices(picks)
      self.forest = self.forest + [tree]
      self.order = order
  def oob_set_generator(self):
    growth_index = len(self.forest)
    index = int(math.floor(uniform(0, 1)*growth_index))
    gen_indices = self.forest[index].get_growth_indices()
    all_indices = range(self.order)
    oob_indices = list(set(all_indices) - set(gen_indices))
    return oob_indices
  def test_predict(self, element, index):
    votes = defaultdict(lambda: 0)
    for tree in self.forest:
      if not tree.contains_seed(index):
        votes[tree.decide_element(element)] += 1
    return max(vote for vote in votes)
  def predict(self, element):
    votes = defaultdict(lambda: 0)
    for tree in self.forest:
      votes[tree.decide_element(element)] += 1
    return max(vote for vote in votes)
