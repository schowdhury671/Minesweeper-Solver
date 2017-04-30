from keras.models import Model, Sequential
#from keras.layers import Convolution2D, Flatten, Dense, Input
from keras import backend as K
import keras.callbacks
from keras.layers import Dense
import numpy as np

class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        #self.acc.append(logs.get('acc'))


class NeuralNet:
    def __init__(self, bsize = (4, 4), gamma = 0.99):
        self.model = Sequential()
        self.model.add(Dense(18, activation='relu', input_dim=18))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(lr=0.0001)
        self.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
        self.model.summary()
        #self.tb = keras.callbacks.TensorBoard(log_dir='/tmp/NN/logs', write_graph=True)
        self.bsize = bsize
        self.gamma = gamma
        self.hist = LossHistory()

    def predict(self, state):
        q = self.model.predict(state,verbose=0)
        return q

    def fit(self, x, y):
        #self.model.train_on_batch(x, y)
        self.model.fit(x, y, epochs=1, callbacks=[self.hist],verbose=0)     #, callbacks=[self.tb])


    def save(self, fpath = 'model.hdf'):
        self.model.save_weights(fpath)

    def load(self, fpath = 'model.hdf'):
        self.model.load_weights(fpath, by_name = False)