
 # selu adam rmse -N
import pandas as pd

data=pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv')
data.head()
d=data['Total Confirmed']
d.values


# evaluate mlp
from math import sqrt
from numpy import array
from numpy import mean
from numpy import std
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in=1, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# fit a model
def model_fit(train, config):
	# unpack config
	n_input, n_nodes, n_epochs, n_batch = config
	# prepare datas
	data = series_to_supervised(train, n_in=n_input)
	train_x, train_y = data[:, :-1], data[:, -1]

	# define model
	model = Sequential()
	model.add(Dense(n_nodes, activation='selu', input_dim=n_input))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	# fit
	history=model.fit(train_x, train_y, epochs=n_epochs,validation_split=0.33, batch_size=n_batch, verbose=1)
	pyplot.plot(history.history['loss'][:])
	pyplot.plot(history.history['val_loss'][:])
	pyplot.title('model train vs validation loss')
	pyplot.ylabel('loss')
	pyplot.xlabel('epoch')
	pyplot.legend(['train', 'validation'], loc='upper right')
	pyplot.show()

	return model

# forecast with a pre-fit model
def model_predict(model, history, config):
	# unpack config
	n_input, _, _, _ = config
	# prepare data
	x_input = array(history[-n_input:]).reshape(1, n_input)
	# forecast
	yhat = model.predict(x_input, verbose=0)
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	#print(' > %.3f' % error)
  
	return error,predictions

# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=1):
	# fit and evaluate the model n times
	scores= [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores




data = d.values
# data split
n_test = 12
# define config[n_input, n_nodes, n_epochs, n_batch ]
config = [24, 500, 200, 110]
# grid search
result= repeat_evaluate(data, config, n_test)
# summarize scores
scores=result[0]

print('mean rmse value for MLP: ',mean(scores[0]))
print('Predicted Next Total Cases In India: ',int(scores[1][-1]))
print('')