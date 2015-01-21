# Abstraction of the Evaluation/Validation process, creates a learner for whom
# various methods of evaluation can be applied. e.g. applying k-fold cross-validation
# to a k-local linear regression model
class EvalWrapper:
	def __init__(self, initial_val, setup_fn):
		self.initial_val = initial_val
		self.setup_fn = setup_fn

	def get_hypothesis(self, size, examples):
		return self.setup_fn(self.initial_val, size, examples)
