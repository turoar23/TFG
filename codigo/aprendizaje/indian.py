from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# load the dataset
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input (z) and output (y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, Y, epochs=150, batch_size=1, verbose=2)
# evaluate the keras model
_, accurary = model.evaluate(X, Y, verbose=2)
print('Accuray: %.2f' % (accurary*100))
