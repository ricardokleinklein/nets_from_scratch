from sklearn import datasets
from sklearn.preprocessing  import normalize

class Iris():
	
	def __init__(self):
		iris = datasets.load_iris()
		self.X = iris.data
		self.Y = iris.target

	def get_data(self):
		return self.X, self.Y

	def get_norm_data(self):
		return normalize(self.X), self.Y
