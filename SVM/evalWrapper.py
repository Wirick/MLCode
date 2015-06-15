# Abstraction of the Evaluation/Validation process, creates a learner for whom
# various methods of evaluation can be applied. e.g. applying k-fold cross-validation
# to a k-local linear regression model
class EvalWrapper:
	def __init__(self, initial_val, setup_fn, examples, lookupfn=lambda x, y:{}):
		self.initial_val = initial_val
		self.setup_fn = setup_fn
		self.examples = examples
		self.lookupfn = lookupfn

	def get_hypothesis(self, size, examples):
		lookup = self.lookupfn(size, examples)
		f = self.setup_fn(self.initial_val, size, examples, lookup)
		return f
