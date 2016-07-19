# This module contains the class RandomForest, as well as the functions
# that a random forest uses to construct it's trees, in particular a way of 
# splitting sets of discrete and continuous attributes.
import random as rnd
from collections import defaultdict
import operator
import math
import decision_tree

def ordered_importance(attributes, examples, predict_index, positive_classes, order):
    """ The ordered importance treats the attributes as ordered with respect to importance
        and splits the data based on that metric. """
    if len(attributes) == 0:
        return [], []
    attr = attributes[0]
    if attr[1] == 'c':
            gini_score, rang = continuous_gini(attr[0], examples, predict_index, positive_classes, order)
    else:
            gini_score, rang = discrete_gini(attr[0], examples, predict_index, positive_classes, order)
    return attr, rang

def discrete_gini(attribute, examples, predict_index, positive_classes, order):
    """ The discrete gini looks for the greatest gini score for each element with respect to
        POSITIVE_CLASSES in a set of EXAMPLES of magnitude ORDER with respect to some PREDICT_INDEX.
        It then takes the top elements and returns a partition comprising that subset and then everything else. """
    order_vals, counts = defaultdict(lambda: 0), defaultdict(lambda: 0)
    all_positive = True
    for ex in examples:
        if ex[predict_index] in positive_classes:
            order_vals[ex[attribute]] += 1
        else: 
            all_positive = False
        counts[ex[attribute]] += 1
    val_order = len(order_vals.items())
    dist_order, dist_set = int(math.floor(math.sqrt(val_order))), []
    vals = defaultdict(lambda: 0)
    for key, value in order_vals.items():
        vals[key] = value / float(counts[key])
    sorted_vals = sorted(vals.items(), key=operator.itemgetter(1), reverse=True)
    side1count, side1pos, side2count, side2pos = 0, 0, 0, 0
    if all_positive:
        yes_split = sorted_vals[:1]
        no_split = sorted_vals[1:]
    else:
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

def continuous_gini(attribute, examples, predict_index, positive_classes, order):
    """ The continuous gini looks at an ATTRIBUTE and decides where to split 
        the EXAMPLES of size ORDER with respect to some POSTIVE_CLASSES and PREDICT_INDEX. """
    ordered = sorted(examples, key=lambda x : float(x[attribute]))
    side2len, side1len, side2pos, side1pos, order = order, 0, 0, 0, order
    gini_max, gini_max_index, last_checked = 0, 0, None
    for x in ordered:
        if x[predict_index] in positive_classes:
            side2pos += 1
    for i in range(1, order):
        consider = ordered[i][predict_index]
        if consider in positive_classes:
            side1pos += 1
            side2pos -= 1
        side1len += 1
        side2len -= 1
        if consider in positive_classes and last_checked:
            continue
        elif consider not in positive_classes and not last_checked:
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

def rf_gini_split(attributes, examples, predict_index, positive_classes, order):
    """ The gini_split function for a RandomForest uses a vector of ATTRIBUTES, and returns the
        best attribute to use to split the EXAMPLES of magnitude ORDER, as well as the variable ranges
        that comprise the partition/continuous range. """
    attr = rnd.sample(attributes, int(math.ceil(math.sqrt(len(attributes)))))
    gini_scores = defaultdict(lambda: 0)
    for attribute in attr:
        if attribute[1] == 'c':
            gini_scores[attribute] = continuous_gini(attribute[0], examples, predict_index, positive_classes, order)
        else:
            gini_scores[attribute] = discrete_gini(attribute[0], examples, predict_index, positive_classes, order)
    split = max([x for x in gini_scores.items()], key=lambda x: float(x[1][0]))
    return split[0], split[1][1]

def rf_attr_fn(attributes, elt):
    """ For random forests, there is no attribute truncation, so the attribute function is the identity
        on the space of elements. """
    return attributes

def wtd_rf_attr_fn(attributes, elt):
    """ To go with the ordered importance fun. """
    lst = list(attributes)
    x = rnd.uniform(0, 1)
    if x < .75:
        return tuple(lst[1:])
    else:
        index = int(math.floor((len(lst)-1)*x))
        return tuple(lst[1:index] + [lst[0]] + lst[index+1:])

# The RandomForest class is a set of decision trees that vote on their choice for 
# a distinguished index in the data.
class RandomForest:
    def __init__(self, examples, trees, mtry, attributes, predict_index, attr_fn, dist_classes, order, imp):
        """ A random forest takes a set of EXAMPLES of size ORDER, and constructs TREES(int), using MTRY number
            of ATTR_FN(ATTRIBUTES) in the split evaluation for each branch with respect to some PREDICT_INDEX. """
        self.forest = set()
        self.trees = trees
        self.importance_fn = imp
        self.order = set(range(order))
        bootstrap_order = int(math.floor(order*2/float(3)))
        for i in range(trees):
            print "Generating tree ", i, ' of order ', bootstrap_order
            picks, growth_indices = set(), set()
            for j in range(bootstrap_order):
                pick = int(math.floor(rnd.uniform(0, 1)*order))
                growth_indices.add(pick)
                dist_elt = tuple(examples[pick])
                picks.add(dist_elt)
            tree = decision_tree.dt_generator(picks, attributes, [], predict_index, imp, attr_fn, dist_classes, len(picks))
            tree.set_growth_indices(growth_indices)
            self.forest.add(tree)

    def oob_set_generator(self):
        """ Used to generate a set of elements from the original dataset, taken from the construction indices
            for some tree, these elements will be ignored by their proginators, so constitute a bootstrap sample
            to used in prediction. This requires the use of test_predict with respect to some original index. """
        dist_tree = rnd.sample(self.forest, 1)
        gen_indices = dist_tree[0].get_growth_indices()
        all_indices = self.order
        oob_indices = set(all_indices - gen_indices)
        return oob_indices

    def test_predict(self, element, index):
        """ When ELEMENT was used in the construction of this random forest with 
            original INDEX, returns the prediction of those trees that were not 
            seeded by this particular element. """
        votes = defaultdict(lambda: 0)
        for tree in self.forest:
            if not tree.contains_seed(index):
                votes[tree.decide_element(element)] += 1
        return max(vote for vote in votes)

    def predict(self, element):
        """ Returns the prediction of this random forest for ELEMENT. """
        votes = defaultdict(lambda: 0)
        for tree in self.forest:
            votes[tree.decide_element(element)] += 1
        return max(vote for vote in votes)
