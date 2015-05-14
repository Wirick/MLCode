from collections import defaultdict

def DecisionTreeLearner(examples, attributes, parent_examples, predict_index, importance):
    if examples == []:
      return DecisionTree((predict_index, 'd'), plurality(parent_examples, predict_index))
    changed, initial = False, examples[0][predict_index]
    for ex in examples[1:]:
      if ex[predict_index] == initial:
        continue
      else:
        changed = True
        break
    if not changed:
      return DecisionTree((predict_index, 'd'), initial)
    if attributes == []:
      return DecisionTree((predict_index, 'd'), plurality(examples, predict_index))
    dist_attr, attr_vals = importance(attributes, examples)
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
      attr_list = list(attributes)
      attr_list.remove(dist_attr)
      subtree = DecisionTreeLearner(dist_examples, attr_list, examples, predict_index, importance)
      tree.add_branch(subtree, val)
    return tree
  
def plurality(examples, index):
    dic = defaultdict(lambda: 0)
    for ex in examples:
      dic[ex[index]] += 1
    return max(x for x in dic)

class DecisionTree:
  children = defaultdict(lambda: 0)
  final = True
  def __init__(self, root_attr, root_vals):
    self.root_attr = root_attr
    self.root_vals = root_vals
  def add_branch(self, branch, label):
    self.final = False
    self.children[label] = branch
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
      branch = self.children[element[self.root_attr[0]]]
      if branch == 0:
        return "This can't be decided by this tree."
      return branch.decide_element(element)
    else:
      return self.root_vals
  def is_leaf(self):
    return self.final
