'''
Implementation of spambase logistic regression problem using a simple one-layer
dense neural network in TensorFlow
'''
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset

from hessian import back_over_back

def loadSpamBase(split, batch_size):
    assert split in ['train', 'test']

    data = pd.read_csv(f'./spambase/spambase_{split}.data', header=None)

    data_map = lambda x: np.log(x+0.1)

    X = data.iloc[:, 0:-1].apply(data_map)
    y = data.iloc[:, -1]

    return Dataset.from_tensor_slices((X, y)).batch(batch_size)

class Model(keras.Model):
    def __init__(self, input_dim):
        super().__init__()

        self.linear = keras.layers.Dense(1, input_shape=input_dim)
        self.sigmoid = keras.layers.Activation('sigmoid')

    '''Custom compile method'''
    # def compile(self, **kwargs):
    #     pass

    def call(self, inputs, training=False):
        x = self.linear(inputs)

        if training:
            return x
        else:
            return self.sigmoid(x)

    '''Custom training method'''
    def train_step(self, data):
        X, y = data

        def closure(self, update):
            #Compute update somehow

            outputs = self(X, training=True)

            return self.compiled_loss(outputs, y)

        grads, hvp = back_over_back(self, X, y)

        self.optimizer.perpare(closure, hvp)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    '''Custom test method'''
    # def test_step(self, data):
    #     pass

def spambase(learn_rate=0.01, batch_size=100, epochs=20):
    #Check for GPU
    if tf.config.list_physical_devices('GPU'):
        print('Using GPU acceleration.')
    else:
        print('Using CPU only.')

    #Build model
    input_dim = 57
    model = Model((input_dim,))

    #Compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learn_rate),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.BinaryAccuracy()])

    #Get datasets
    train = loadSpamBase('train', batch_size)
    test = loadSpamBase('test', batch_size)

    print('Starting to train...')

    model.fit(train, epochs=epochs)

    print('Training finished, now evaluating...')

    model.evaluate(test)

if __name__=='__main__':
    spambase()
