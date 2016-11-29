import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils



def dataGenerator(X_train, Y_train, batch_size):
    assert not (len(Y_train)%batch_size), 'batch_size is not a factor of no of instances'

    no_of_batches = int(len(Y_train)/batch_size)
    while 1:
        for i in range(no_of_batches): # 1875 * 32 = 60000 -> # of training samples
            if i%125==0:
                print "i = " + str(i)
            yield X_train[i*batch_size:(i+1)*batch_size], Y_train[i*batch_size:(i+1)*batch_size]


def sample_code():
    batch_size = 32
    test_batch_size = 1
    nb_classes = 10
    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)


    #some examples of the training data
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(X_train[i], cmap='gray', interpolation='none')
        plt.title("Class {}".format(y_train[i]))

    #Our neural-network is going to take a single vector for each training example, so we need to reshape the input so that each 28x28 image becomes a single 784 dimensional vector. We'll also scale the inputs to be in the range [0-1] rather than [0-255]
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)

    #Modify the target matrices to be in the one-hot format
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    #Build the neural-network. Here we'll do a simple 3 layer fully connected network

    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))  # An "activation" is just a non-linear function applied to the output
    # of the layer above. Here, with a "rectified linear unit",
    # we clamp all values below 0 to 0.

    model.add(Dropout(0.2))  # Dropout helps protect the model from memorizing or "overfitting" the training data
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))  # This special "softmax" activation among other things,
    # ensures the output is a valid probaility distribution, that is
    # that its values are all non-negative and sum to 1.

    #loss and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    #feed the training data loaded in earlier into this model
    # model.fit(X_train, Y_train,
    #           batch_size=128, nb_epoch=4,
    #           show_accuracy=True, verbose=1,
    #           validation_data=(X_test, Y_test))

    generator = dataGenerator(X_train, Y_train, batch_size)
    model.fit_generator(generator, len(Y_train), nb_epoch=1, verbose=1, callbacks=[], validation_data=(X_test, Y_test))
    #score = model.evaluate(X_test, Y_test,
    #                       show_accuracy=True, verbose=0)
    #evaluate_generator(self, generator, val_samples, max_q_size=10, nb_worker=1, pickle_safe=False)
    testgenerator = dataGenerator(X_test, Y_test, test_batch_size)
    score = model.evaluate_generator(testgenerator, len(Y_test))
    #print('test loss', score)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    '''
    #Inspecting the output

    # The predict_classes function outputs the highest probability class
    # according to the trained classifier for each input example.
    predicted_classes = model.predict_classes(X_test)

    # Check which items we got right / wrong
    correct_indices = np.nonzero(predicted_classes == y_test)[0]
    incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
    '''

    return


def overfit_model(train_generator, test_generator):
    
    return

def train_model():
    return
if __name__ == "__main__":
    sample_code()

'''
#For analysing images that were correctly and incorrectly classified
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
'''