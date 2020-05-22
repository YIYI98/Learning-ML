# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 10:31:40 2018

@author: Wendong Zheng
"""

from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_excel
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree


np.random.seed(1337)  # for reproducibility
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
# load dataset
dataset = read_excel('42Merge.xlsx', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
# values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[13,14,15,16,17,18,19,20,21,22,23]], axis=1, inplace=True)
print(reframed.head())
# split into train and test sets
values = reframed.values
n_train_hours = 20000
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
test_X1, test_y1 = test[:, :-1], test[:, -1]
test_X2, test_y2 = test[:, :-1], test[:, -1]
test_X3, test_y3 = test[:, :-1], test[:, -1]
test_X4, test_y4 = test[:, :-1], test[:, -1]
test_X5, test_y5 = test[:, :-1], test[:, -1]
#SVM Train
clf = SVR()
clf.fit(train_X,train_y)

# design KNN model
clf2 = KNeighborsRegressor(n_neighbors=5)
# fit model
clf2.fit(train_X,train_y)

# design dtree model
clf3 = tree.DecisionTreeRegressor()
# fit model
clf3.fit(train_X,train_y)


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
test_X1 = test_X1.reshape((test_X1.shape[0], 1, test_X1.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



#1 layer
# design network LSTM
print('Build LSTM model...')
model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# design network TG-LSTM
print('Build Our model...')
model1 = Sequential()
model1.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]),implementation=2))
model1.add(Dense(1))
model1.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history1 = model1.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X1, test_y1), verbose=2, shuffle=False)



# plot history train-loss
pyplot.ylabel("Train loss value")
pyplot.xlabel("The number of epochs")
pyplot.title("Loss function-epoch curves")
pyplot.plot(history.history['loss'], label='train_LSTM')
pyplot.plot(history1.history['loss'], label='train_TG-LSTM')
pyplot.legend()
pyplot.savefig('Figure-Flood-train-loss.png', dpi=300)
pyplot.show()

# plot history val-loss
pyplot.ylabel("Validation Loss value")
pyplot.xlabel("The number of epochs")
pyplot.title("Loss function-epoch curves")
pyplot.plot(history.history['val_loss'], label='val_LSTM')
pyplot.plot(history1.history['val_loss'], label='val_TG-LSTM')
pyplot.legend()
pyplot.savefig('Figure-Flood-val-loss.png', dpi=300)
pyplot.show()

# make a prediction LSTM
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# make a prediction LSTM-Our
yhat1 = model1.predict(test_X1)
test_X1 = test_X1.reshape((test_X1.shape[0], test_X1.shape[2]))

# make a prediction SVM

predict_y = clf.predict(test_X)
yhat2 = predict_y.reshape(predict_y.shape[0],1)

# make a prediction KNN

predict_y2= clf2.predict(test_X)
yhat3= predict_y2.reshape(predict_y2.shape[0],1)

# make a prediction tree

predict_y3= clf3.predict(test_X)
yhat4= predict_y3.reshape(predict_y3.shape[0],1)



# invert scaling for forecast LSTM
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for forecast LSTM-Our
inv_yhat1 = concatenate((yhat1, test_X1[:, 1:]), axis=1)
inv_yhat1 = scaler.inverse_transform(inv_yhat1)
inv_yhat1 = inv_yhat1[:,0]

# invert scaling for forecast SVM
inv_yhat2 = concatenate((yhat2, test_X2[:, 1:]), axis=1)
inv_yhat2 = scaler.inverse_transform(inv_yhat2)
inv_yhat2 = inv_yhat2[:,0]

# invert scaling for forecast KNN
inv_yhat3 = concatenate((yhat3, test_X3[:, 1:]), axis=1)
inv_yhat3 = scaler.inverse_transform(inv_yhat3)
inv_yhat3 = inv_yhat3[:,0]

# invert scaling for forecast tree
inv_yhat4 = concatenate((yhat4, test_X4[:, 1:]), axis=1)
inv_yhat4 = scaler.inverse_transform(inv_yhat4)
inv_yhat4 = inv_yhat4[:,0]



# invert scaling for actual LSTM
inv_y = scaler.inverse_transform(test_X)
inv_y = inv_y[:,0]

# invert scaling for actual LSTM-Our
inv_y1 = scaler.inverse_transform(test_X1)
inv_y1 = inv_y1[:,0]

# invert scaling for actualSVM
inv_y2 = scaler.inverse_transform(test_X)
inv_y2 = inv_y2[:,0]

# calculate RMSE and MAE LSTM
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mae = mean_absolute_error(inv_y, inv_yhat)
print('LSTM Test RMSE: %.3f' % rmse)
print('LSTM Test MAE: %.3f' % mae)

# calculate RMSE and MAE Our
rmse1 = sqrt(mean_squared_error(inv_y1, inv_yhat1))
mae1 = mean_absolute_error(inv_y1, inv_yhat1)
print('Our method Test RMSE: %.3f' % rmse1)
print('Our method Test MAE: %.3f' % mae1)

pyplot.figure(figsize=(20,10))
pyplot.title('Runoff1000')
pyplot.xlabel('Time range(h)')
pyplot.ylabel(' Runoff range')
pyplot.plot(inv_y[800:1000],label='true')
pyplot.plot(inv_yhat[800:1000],'r--',label='predictions_LSTM')
pyplot.plot(inv_yhat1[800:1000],'g--',label='predictions_TG-LSTM')
pyplot.plot(inv_yhat2[800:1000],'c-.',label='predictions_SVM')
# pyplot.plot(inv_yhat3[800:1000],'k-.',label='predictions_KNN')
# pyplot.plot(inv_yhat4[800:1000],'y-.',label='predictions_DTree')
pyplot.legend()
pyplot.savefig('Tunxi Predict1h-lstm-tg-svm', dpi=300)
pyplot.show()