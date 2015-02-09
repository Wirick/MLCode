import util
# Class for wrapping up a lookup function so that memoization
# can be incorperated.
class LookupWrapper:
	
	def __init__(self, lookup, metric, magnitude, reduction_function):
		lookupfn = lambda x, y: lookup(x, y, metric)
		self.lookup = util.memoized(lookupfn, lambda x: tuple(x))
		self.metric = metric
		self.magnitude = magnitude
		self.reduction_function = reduction_function

	def lookup(self, size, examples):
		dic = self.lookup(self.magnitude, examples, self.metric)
		for key in dic:
			dic[key] = self.reduction_function(size, dic[key])
		return dic
