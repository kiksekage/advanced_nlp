from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

#maybe use the best model as the starting point and
#try to see if varying one param at a time chenges anything
#this way we could at least investigate the influence of each param in isolation
class Seq2Seq:
	"""
	this class provides the functionalities needed to create a model given parameters,
	train and predict commands given sentences;
	Init Arguments:
		net_type (str): SRN, LSTM or GRU
		num_layers (int): either 1 or 2
		hidden_size (int): 25, 50, 100, 200 or 400
		dropout (float): specifies the dropout rate, can be 0, 0.1 or 0.5
		attention (bool): determines if attention should be used
	"""

	def __init__(self, net_type, num_layers, hidden_size, dropout, attention):
		self.model = None
		if net_type == 'SRN':
			#self.model = DEFINE SRN
		if net_type == 'LSTM':
			#self.model = DEFINE LSTM
		if net_type == 'GRU':
			#self.model = DEFINE GRU

	def fit(self):
		pass

	def predict(self):
		pass


class GridSearch:
	"""
	WARNING: DON"T USE IT UNLESS YOU KNOW YOUR IMPLEMENTATION WORKS CORRECTLY, I.E. TRAINS 
	SUCCESSFULLY WITH THE BEST PARAMETERS AND ACHIEVES RESULTS CLOSE TO THE ONES FROM THE PAPER
	
	Provides the functionalities needed to perform a restricted grid search, given parameters
	of the best model, as written in the paper.
	Grid search is performed by starting from the best parameters and trying different models
	by varying one parameter at a time. In other words all but one parameter are the same in the
	best model. This gives 12, instead of 180, different models for each experiment.
	Each model is trained 5 time with random initialization, and the mean score is reported.
	This gives 60 training sessions for each experiment.
	This function is meant to handle a single experiment, e.g. in 1a we could run the grid search
	to train the best model and try to vary one parameter at a time to confirm that it doesn't
	improve the performance.
	But of course, this should be usable for each experiment
	
	Init Arguments:
		Provide the best parameters for the given experiment
		best_net_type (str): SRN, LSTM or GRU
		best_num_layers (int): either 1 or 2
		best_hidden_size (int): 25, 50, 100, 200 or 400
		best_dropout (float): specifies the dropout rate, can be 0, 0.1 or 0.5
		best_attention (bool): determines if attention should be used
	"""

	def __init__(best_net_type, best_num_layers, best_hidden_size, best_dropout, best_attention):
		#stores accuracies of all models to be trained
		#each entry stores the results of varying this parameer, i.e. all other params are the same as in the best model
		#each value is a k x 5 matrix, where k is the number of possible values of the parameter and 5, as every model is trained 5 times
		self.results = {
			'net_type': np.zeros((3, 5)),
			'num_layers': np.zeros((2, 5)),
			'dropout': np.zeros((3, 5)),
			'hidden_size': np.zeros((5, 5)),
			'attention': np.zeros((2, 5))
		}

		self.num_layers = {'num_layers': [1, 2]}
		self.hidden_size = {'hidden_size': [25, 50, 100, 200, 400]}
		self.dropout = {'dropout': [0, 0.1, 0.5]}
		self.attention = {'attention': [True, False]}
		self.net_type = {'net_type': ['srn', 'lstm', 'gru']}
		
		#parameters of the best model for the given experiment
		self.best_net_type = best_net_type
		self.best_num_layers = best_num_layers
		self.best_hidden_size = best_hidden_size
		self.best_dropout = best_dropout
		self.best_attention = best_attention

	def train_eval(self):
		"""
		Trains and evaluates all the models, stores the results
		in self.results
		"""
		pass