from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, ZeroPadding2D, \
    MaxPooling2D, Activation, Dense, Dropout, Flatten
import h5py, os
import subprocess, traceback


class Utility:
    # Template command for converting caffe model to a keras
    # sequential model. Do not forget to replace MODELPATH,
    # PROTONAME, OUTPATH
    caffe2kerascmd = 'python -m ./keras/keras/caffe/caffe2keras.py ' \
                     '-load_path MODELPATH -prototxt PROTONAME ' \
                     '-caffemodel VGG_ILSVRC_16_layers ' \
                     '-store_path OUTPATH -network_type ' \
                     'Sequential'

    def __init__(self):
        return

    def create_caffe_model_command(self, model_path='/tmp',
                                   output_path='/tmp',
                                   proto_path='/tmp'):
        '''
        This method prepares the model conversion from caffe to keras
        :param model_path: name of the database
        :param output_path: Path to dump the keras weights
        :param proto_path: Path to the proto text
        :return: command string
        '''
        cmd = self.caffe2kerascmd.replace("MODELPATH", model_path)
        cmd = cmd.replace("PROTONAME", proto_path)
        cmd = cmd.replace("OUTPATH", output_path)
        return cmd

    def convertCaffeModel2Keras(self, model_path='/tmp',
                                output_path='/tmp',
                                proto_path='/tmp'):
        '''
        This method converts the caffe model to keras model
        :param model_path: name of the database
        :param output_path: Path to dump the keras weights
        :param proto_path: Path to the proto text
        :return: The output file path if success else None
        '''
        try:
            command = self.create_caffe_model_command(model_path,
                                                      output_path,
                                                      proto_path)
            status = subprocess.call(command.split(" "))
            if status is 0:
                return output_path
            else:
                print("Failed to load caffe model " +
                               str(model_path))
                return None
        except Exception as e:
            print(
                "Exception occurred while loading caffe model"
                "Stacktrace: \n" +
                traceback.format_exc())
            return False

    def getVggModel(self, model_path='/tmp'):
        '''
        This method returns a pretrained Keras VGG Model
        :param model_path: name of the database
        :param output_path: Path to dump the keras weights
        :param proto_path: Path to the proto text
        :return: Keras Model object
        '''

        model_weights = os.path.join(model_path,
                                     'Keras_model_weights.h5')

        model = self.build_vgg_model()
        f = h5py.File(model_weights)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected)
                # layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in
                       range(g.attrs['nb_params'])]
            try:
                model.layers[k].set_weights(weights)
            except Exception:
                pass
        f.close()
        return model

    # How to load the model
    def build_vgg_model(img_width=224, img_height=224):

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, 224,
                                                     224)))
        model.add(Convolution2D(64, 3, 3, activation='relu',
                                name='conv1_1'))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu',
                                name='conv1_2'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu',
                                name='conv2_1'))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu',
                                name='conv2_2'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu',
                                name='conv3_1'))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu',
                                name='conv3_2'))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu',
                                name='conv3_3'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu',
                                name='conv4_1'))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu',
                                name='conv4_2'))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu',
                                name='conv4_3'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu',
                                name='conv5_1'))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu',
                                name='conv5_2'))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu',
                                name='conv5_3'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000))
        model.add(Activation('softmax'))
        return model
