from collections import defaultdict
import operator

def dt_generator(examples, attributes, parent_examples, predict_index, importance, attr_fn, positive_classes):
    if examples == []:
      return DecisionTree((predict_index, 'd'), plurality(parent_examples, predict_index))
    changed, initial = False, examples[0][predict_index]
    for ex in examples:
      if ex[predict_index] == initial:
        continue
      else:
        changed = True
        break
    if not changed:
      return DecisionTree((predict_index, 'd'), initial)
    if attributes == []:
      return DecisionTree((predict_index, 'd'), plurality(examples, predict_index))
    if len(examples) < 100:
      return DecisionTree((predict_index, 'd'), plurality(examples, predict_index))
    dist_attr, attr_vals = importance(attributes, examples, predict_index, positive_classes)
    tree = DecisionTree(dist_attr, attr_vals)
    for val in attr_vals:
      dist_examples = []
      if dist_attr[1] == 'c':
        for e in examples:
          if float(e[dist_attr[0]]) >= float(val[0]) and float(e[dist_attr[0]]) < float(val[1]):
            dist_examples = dist_examples + [e]
      else: 
        for e in examples:
          if e[dist_attr[0]] == val:
            dist_examples = dist_examples + [e]
      attr_list = attr_fn(attributes, dist_attr)
      subtree = dt_generator(dist_examples, attr_list, examples, predict_index, importance, attr_fn, positive_classes)
      tree.add_branch(subtree, val)
    return tree
  
def plurality(examples, index):
    dic = defaultdict(lambda: 0)
    for ex in examples:
      dic[ex[index]] += 1
    arg_max = max([x for x in dic.items()], key=operator.itemgetter(1))
    return arg_max[0]

class DecisionTree:
  children = defaultdict(lambda: 0)
  growth_indices = []
  final = True
  def __init__(self, root_attr, root_vals):
    if len(root_vals) == 2:
      if root_vals[1] == ():
        root_vals = root_vals[1]
    self.root_attr = root_attr
    self.root_vals = root_vals
  def add_branch(self, branch, label):
    self.final = False
    self.children[label] = branch
  def set_growth_indices(self, lst):
    self.growth_indices = lst
  def get_growth_indices(self):
    return self.growth_indices
  def contains_seed(self, elt):
    return elt in self.growth_indices
  def decide_element(self, element):
    if not self.is_leaf():
      if self.root_attr[1] == 'c':
        for val in self.root_vals:
          if val[0] == val[1] and float(element[self.root_attr[0]]) == float(val[0]):
            branch = self.children[val]
            return branch.decide_element(element)
          vals = (float(val[0]), float(val[1]))
          if vals[0] <= element[self.root_attr[0]]:
            if float(element[self.root_attr[0]]) < vals[1]:
              branch = self.children[val]
              return branch.decide_element(element)
        if float(element[self.root_attr[0]]) < self.root_vals[0][0]:
          branch = self.children[self.root_vals[0]]
          return branch.decide_element(element)
        else:
          branch = self.children[self.root_vals[len(self.root_vals) - 1]]
          return branch.decide_element(element)
      found, branch = False, None
      for key in self.children:
        if element[self.root_attr[0]] in key:
          branch = self.children[key]
          found = True
          break
      if not found:
        branch = self.children[element[self.root_attr[0]]]
      if branch == 0:
        return 'This cant be decided by this tree'
      return branch.decide_element(element)
    else:
      return self.root_vals
  def is_leaf(self):
    return self.final
