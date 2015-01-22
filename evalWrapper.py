# Abstraction of the Evaluation/Validation process, creates a learner for whom
# various methods of evaluation can be applied. e.g. applying k-fold cross-validation
# to a k-local linear regression model
class EvalWrapper:
	def __init__(self, initial_val, setup_fn, examples, lookup):
		self.initial_val = initial_val
		self.setup_fn = setup_fn
		print 'beginning lookup computation'
		self.lookup = lookup(initial_val, examples, True)
		print 'lookup computation finished'
		self.examples = examples

	def get_hypothesis(self, size, examples):
		examples = [ex for ex in self.examples if ex in examples]
		f = memoized(self.setup_fn(self.initial_val, size, examples, self.lookup), lambda x: str(x))
		return f

