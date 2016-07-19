# This skeleton file can be found at 
# https://www.cs.berkeley.edu/~russell/classes/cs194/f11/assignments/a4/NBmodel.py
# it was used for an experimenal machine learning class at berkeley: CS 194
import pickle
import os
import re
import math
from collections import defaultdict

#########################################

def generate_Boolean_bayes(input_dirs, features_file, model_file): 
    """ Given INPUT_DIRS, a FEATURES_FILE, munges the content of the input_dirs
        and then returns a Naive Bayes classifier. """
    labels = []
    nb_model, probs = defaultdict(lambda: 0), defaultdict(lambda: 0)
    features = pickle.load(open(features_file, 'rb'))
    for directory in input_dirs:
      print "Generating for dir", directory[6:]
      label = directory[6:]
      labels = labels + [label]
      for f in get_files(directory):
        nb_model[label] += 1
        probs['order'] += 1
        munge = munge_Boolean(f, features)
        for i in range(len(features)):
          if munge[i] > 0:
            nb_model[(i, label)] += 1
            probs[i] += 1
    for i in range(len(features)):
      for lab in labels:
        if nb_model[(i, lab)] > 0:
          nb_model[(i, lab)] = math.log(nb_model[(i, lab)]/float(probs[i]))
    for lab in labels:
      nb_model[lab] = math.log(nb_model[lab]/float(probs['order']))
    dic = {}
    dic.update(nb_model)
    pickle.dump(dic, open(model_file, 'w'))
    return NB_Boolean(features_file, model_file)

def generate_NTF_bayes(input_dirs, features_file, model_file):
    """ Similar to the above function, except returns a Bayes Classifer using
        Normalized term frequency on the data. """
    labels = []
    nb_model, probs = defaultdict(lambda: 0), defaultdict(lambda: 0)
    features = pickle.load(open(features_file, 'rb'))
    feat_order = len(features)
    for directory in input_dirs:
      print "Generating for dir", directory[6:]
      label = directory[6:]
      labels = labels + [label]
      for f in get_files(directory):
        nb_model[label] += 1
        probs['order'] += 1
        munge = munge_NTF(f, features)
        for i in range(feat_order):
          if munge[i] > 0:
            nb_model[(i, label)] += munge[i]
             probs[i] += 1
    for i in range(feat_order):
      for lab in labels:
        if nb_model[(i, lab)] > 0:
          nb_model[(i, lab)] = nb_model[(i, lab)]/float(probs[i])
    for lab in labels:
      nb_model[lab] = math.log(nb_model[lab]/float(proba['order']))
    dic = {}
    dic.update(nb_model)
    pickle.dump(dic, open(model_file, 'w'))
    return NB_NTF(features_file, model_file)   
  
def munge_Boolean(email_file,features):
    """ Returns a tuple of booleans, where x[i]=1 if the EMAIL_FILE
        contains the word represented by features[i], and 0 otherwise. """
    tokens = set([token for token in re.split(' |\r\n', open(email_file).read())])
    bool_munge = [0 for x in features]
    for token in tokens:
      if token not in features:
        continue
      index = features.index(token)
      if index > -1:
        bool_munge[index] = 1
    return tuple(bool_munge)

def munge_NTF(email_file, features):
    """ Returns a tuple of floating point numbers where x[i]=c represents
        the normalized term frequency of FEATURES[i] in the EMAIL_FILE. """
    tokens = [token for token in re.split(' |\r\n', open(email_file).read())]
    order, pos_features = len(tokens), defaultdict(lambda: 0)
    features = pickle.load(open(features, 'rb'))
    ntf_munge = [0 for x in features]
    for token in tokens:
      pos_features[token] += 1
    for token, magnitude in pos_features.items():
      index = features.index(token)
      if index > -1:
        ntf_munge[index] = magnitude/float(order)
    return tuple(ntf_munge)
    

def NBclassify_Boolean(example, model, cost_ratio):
    max_label, max_score, cost_ratio = None, -float('inf'), math.log(cost_ratio)
    labels = ['spam', 'ham']
    for label in labels:
      current_score = model[label]
      for i in range(len(example)):
        if example[i] == 1:
          current_score += model[(i, label)]
          if label == 'ham':
            current_score += cost_ratio
      if max_score < current_score:
        max_score = current_score
        max_label = label
    return 1 if max_label == 'spam' else 0
      

def NBclassify_NTF(example,model,cost_ratio):
    return 0


#########################################


def get_files(path):
    for f in os.listdir(path):
        f = os.path.abspath( os.path.join(path, f ) )
        if os.path.isfile( f ):
            yield f


class NaiveBayesModel:
    
    def __init__(self, features_file, model_file):
        self.features = pickle.load(open(features_file,'rb'))
        self.model = pickle.load(open(model_file,'rb'))

    def test(self, spam_dir, ham_dir, cost_ratio):
        N = 0
        loss = 0.
        for f in get_files(spam_dir):
            N += 1
            classification = self.classify(self.munge(f),cost_ratio)
            if not (classification==1):
                loss += 1
    
        for f in get_files(ham_dir):
            N += 1
            classification = self.classify(self.munge(f),cost_ratio)
            if not (classification==0):
                loss += cost_ratio
        
        print "Classifier average loss: %f" % (loss/N)


class NB_Boolean(NaiveBayesModel):
    
    def classify(self,example,cost_ratio):
        return NBclassify_Boolean(example,self.model,cost_ratio)
        
    def munge(self,email_file):
        return munge_Boolean(email_file,self.features)


class NB_NTF(NaiveBayesModel):    
    
    def classify(self,example,cost_ratio):
        return NBclassify_NTF(example,self.model,cost_ratio)
    
    def munge(self,email_file):
        return munge_NTF(email_file,self.features)


#########################################


model = generate_Boolean_bayes(['train/ham', 'train/spam'], 'token_list.pkl', 'nb_spam_model.pkl') 
model = NB_Boolean('token_list.pkl', 'nb_spam_model.pkl')
model.test('train/ham', 'train/spam', 1)
