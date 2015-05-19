from collections import defaultdict
import random
import operator
# Generates a decision tree with given ORDER and EXAMPLES as data. ATTRIBUES is a vector of (int, {'c', 'd'}) describing
# the attributes used to construct the tree using the IMPORTANCE function and a PREDICT_INDEX. The importance
# function decides which of a subset of attributes (given by the ATTR_FN) to use to split the data given a set
# of POSITIVE_CLASSES.
def dt_generator(examples, attributes, parent_examples, predict_index, importance, attr_fn, positive_classes, order):
    if not examples:
      return DecisionTree((predict_index, 'd'), plurality(parent_examples, predict_index))
    changed, initial = False, None
    for ex in examples:
      if initial == None:
        initial = ex[predict_index]
        continue
      elif ex[predict_index] == initial:
        continue
      else:
        changed = True
        break
    if not changed:
      return DecisionTree((predict_index, 'd'), initial)
    if len(attributes) == 0:
      return DecisionTree((predict_index, 'd'), plurality(examples, predict_index))
    dist_attr, attr_vals = importance(attributes, examples, predict_index, positive_classes, order)
    tree = DecisionTree(dist_attr, attr_vals)
    for val in attr_vals:
      dist_examples = set()
      if dist_attr[1] == 'c':
        for e in examples:
          if float(e[dist_attr[0]]) >= float(val[0]) and float(e[dist_attr[0]]) < float(val[1]):
            dist_examples.add(e)
      else:
        for e in examples:
          if e[dist_attr[0]] in val:
            dist_examples.add(e)
      attr_list = attr_fn(attributes, dist_attr)
      new_order = len(dist_examples)
      subtree = dt_generator(dist_examples, attr_list, examples, predict_index, importance, attr_fn, positive_classes, new_order)
      tree.add_branch(subtree, val)
    return tree  

# Returns the most voted ex[INDEX] for ex in EXAMPLES.
def plurality(examples, index):
    dic = defaultdict(lambda: 0)
    for ex in examples:
      dic[ex[index]] += 1
    arg_max = max([x for x in dic.items()], key=operator.itemgetter(1))
    return arg_max[0]

# A decision tree has a ROOT_ATTRibute, corresponding to a variable
# x_i value, and root_vals, corresponding to the range of the value 
# of x_j, where x_i mostly, but not always, not equal to x_j.
class DecisionTree:
  # Assigns my root as ROOT_ATTR and it's values as ROOT_VALS
  def __init__(self, root_attr, root_vals):
    self.growth_indices = []
    self.final = True
    self.children = defaultdict(lambda: 0)
    self.root_attr = root_attr
    self.root_vals = root_vals
  # Adds a BRANCH(DecisionTree) whose value of branch.root_attr is LABEL
  def add_branch(self, branch, label):
    self.final = False
    self.children[label] = branch
  # Sets the set of indices of the generating set of this tree as LST. Used
  # so that element_i predictions in the original dataset will not be evaluated
  # by trees that were grown using element_i.
  def set_growth_indices(self, lst):
    self.growth_indices = lst
  # Returns the set of growth indices of this tree.
  def get_growth_indices(self):
    return self.growth_indices
  # Returns true exactly when ELT(int) was used to generate this tree.
  def contains_seed(self, elt):
    return elt in self.growth_indices
  # Returns the prediction of this tree for ELEMENT.
  def decide_element(self, element):
    if not self.is_leaf():
      if self.root_attr[1] == 'c':
        for val in self.root_vals:
          if val[0] == val[1] and float(element[self.root_attr[0]]) == float(val[0]):
            branch = self.children[val]
            return branch.decide_element(element)
          vals = (float(val[0]), float(val[1]))
          if vals[0] <= float(element[self.root_attr[0]]):
            if float(element[self.root_attr[0]]) < vals[1]:
              branch = self.children[val]
              return branch.decide_element(element)
        if float(element[self.root_attr[0]]) < self.root_vals[0][0]:
          branch = self.children[self.root_vals[0]]
          return branch.decide_element(element)
        else:
          branch = self.children[self.root_vals[len(self.root_vals) - 1]]
          return branch.decide_element(element)
      found, branch, val = False, None, None
      for key in self.children:
        if element[int(self.root_attr[0])] in key:
          branch = self.children[key]
          val = key
          found = True
          break
      if not found:
        branch = self.children[random.sample(self.children, 1)[0]]
      return branch.decide_element(element)
    else:
      return self.root_vals
  # Returns true exactly when this tree has no children
  def is_leaf(self):
    return self.final
