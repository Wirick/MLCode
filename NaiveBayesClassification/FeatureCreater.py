from NBmodel import get_files
from collections import defaultdict
import pickle
import re
class FeatureGenerator:

  def __init__(self, tokenized_input_dirs):
    self.tokenized_input_dirs = tokenized_input_dirs
    self.truncation_function = lambda x: x

  def generate_all_features(self):
    """ Generate a feature file based on the tokenized input directory.
        Essentially create a dictionary based on the tokens in the data. """
    all_tokens = set()
    for directory in self.tokenized_input_dirs:
      print "Generating for dir", directory
      for f in get_files(directory):
        tokens = set([token for token in re.split(' |\r\n', open(f).read())])
        all_tokens = all_tokens | tokens
    pickle.dump(list(all_tokens), open('token_list.pkl', 'w'))
    print 'Dumped features'

  def generate_popular_features(self, top_n=10000, per_file_features=50):
    """ Only use the TOP_N features for a dataset, and up to PER_FILE_FEATURES
        for an individual file."""
    all_tokens = set()
    for directory in self.tokenized_input_dirs:
      print "Generating for directory", directory
      intermediate_tokens = set()
      for f in get_files(directory):
        dir_pop = defaultdict(int)
        tokens = [token for token in re.split(' |\r\n', open(f).read())]
        for token in tokens:
          dir_pop[token] += 1
        ordered = sorted(dir_pop.items(), key=lambda x: x[1], reverse=True)
        if len(ordered) > per_file_features:
          ordered = ordered[:per_file_features]
        ordered = set([x[0] for x in ordered])
        intermediate_tokens = intermediate_tokens | ordered
      intermediate_tokens = sorted(list(intermediate_tokens), key=lambda x: dir_pop[x], reverse=True)
      if len(intermediate_tokens) > top_n:
        intermediate_tokens = intermediate_tokens[:top_n]
      intermediate_tokens = set(intermediate_tokens)
      all_tokens = (all_tokens - intermediate_tokens) | (intermediate_tokens - all_tokens) 
    pickle.dump(list(all_tokens), open('token_list.pkl', 'w'))
        

  def set_feature_truncation(self, truncation_fn):
    """ Set a TRUNCATION_FN which should be used to change the feature set. """
    self.truncation_fn = truncation_fn

generator = FeatureGenerator(['train/ham', 'train/spam'])
generator.generate_popular_features()
print 'hello'
