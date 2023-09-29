from argparse import ArgumentParser
import h5py
# import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
# from keras.layers.convolutional import Conv3D
from keras.layers import Conv3D
# from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import ConvLSTM2D
# from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization
# import numpy as np
# import pylab as plt
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping,ModelCheckpoint
# import imageio
# from pathlib import Path
# from sklearn import preprocessing
# import h5py
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Flatten,Input
# from keras.layers.merge import concatenate,add
from keras.layers import concatenate,add
from keras.models import Model
from LOMUQ_sequencer import LOMUQ_sequencer, numpy_normalize
import traceback
import numpy as np
import pickle

def normalize(t):
    t_normalize=np.zeros(t.shape, t.dtype)
    oldmin=t.min()
    oldmax=t.max()
    oldrange=oldmax-oldmin
    for i in t.shape[0]:
        t_normalize[i]=(t[i]-oldmin)/oldrange
    return t_normalize


if __name__ == '__main__':
    description = ("Creating model Bayesian ConvLSTM network and its training routine " +
                   "for predicting Lagrangian ocean particles\n\n" +
                   "Notice!! The training and validation now needs separate " +
                   "pickle files. See arguments: -t, -T, -v, and -V.")
    option_parser = ArgumentParser(description=description)

    r_help = ("path to data folder with results (i.e. parent of output- and " +
              "model folder)")
    option_parser.add_argument("-r", "--resultpath",
                               action="store", dest="resultpath",
                               default="../data", help=r_help)

    m_help = ("name of the model file")
    option_parser.add_argument("-m", "--modelname",
                               action="store", dest="modelname",
                               default="reCT_epoch_{epoch:04d}", help=m_help)

    tdi_help = "training input data directory"
    option_parser.add_argument("-T", "--train-dir-in",
                               action="store", dest="train-dir",
                               default="../data/train", help=tdi_help)

    vdi_help = "validation input data directory"
    option_parser.add_argument("-V", "--valid-dir-in",
                               action="store", dest="valid-dir",
                               default="../data/train", help=vdi_help)

    options = option_parser.parse_args()
    arg_dict = vars(options)

    in_dir_path = "/media/christian/DATA/data/LOMUQ/in/"
    resultpath = "/media/christian/DATA/data/LOMUQ/out/"
    dir_path = "/media/christian/DATA/data/LOMUQ/processed/"

    modelfilename = arg_dict["modelname"]
    mhistfile = os.path.join(resultpath, "output", modelfilename + "_Thist.pkl")
    weightfile = os.path.join(resultpath, "models", modelfilename + "_weights.h5")
    modelfile = os.path.join(resultpath, "models", modelfilename + "_model.h5")

    try:
        particleCountListFile = h5py.File(str(dir_path) + "/particleCountList.h5", 'r')
        hydrodynamic_U_dataListFile = h5py.File(str(dir_path) + "/hydrodynamic_U_dataList.h5", 'r')
        hydrodynamic_V_dataListFile = h5py.File(str(dir_path) + "/hydrodynamic_V_dataList.h5", 'r')

        particleCountList = particleCountListFile['ParticleCount'] # [()]
        hydrodynamic_V_dataList = hydrodynamic_V_dataListFile['hydrodynamic_V'] # [()]
        hydrodynamic_U_dataList = hydrodynamic_U_dataListFile['hydrodynamic_U'] # [()]

        # ================================================================== #
        # hydrodynamic_U_dataList = normalize(hydrodynamic_U_dataList)
        # hydrodynamic_V_dataList = normalize(hydrodynamic_V_dataList)
        # ================================================================== #
        # hydrodynamic_U_dataList = numpy_normalize(hydrodynamic_U_dataList)
        # hydrodynamic_V_dataList = numpy_normalize(hydrodynamic_V_dataList)
        # ================================================================== #

        assert hydrodynamic_V_dataList.shape[0] > 3
        n_train_samples = hydrodynamic_V_dataList.shape[0] - 1 - 3
        mapsize = (hydrodynamic_V_dataList.shape[2], hydrodynamic_V_dataList.shape[3])
        n_scenarios = hydrodynamic_V_dataList.shape[0]
        n_steps = hydrodynamic_V_dataList.shape[1]
        epoch_steps = 10
        batch_size = 4  # 32


        # train_V = hydrodynamic_V_dataList[:n_train_samples]
        # train_U = hydrodynamic_U_dataList[:n_train_samples]
        # train_Y = particleCountList[:n_train_samples]
        # test_V = hydrodynamic_V_dataList[n_train_samples:]
        # test_U = hydrodynamic_U_dataList[n_train_samples:]
        # test_Y = particleCountList[n_train_samples:]
        # if len(test_Y.shape) <= 3:
        #     test_U = np.expand_dims(test_U, axis=0)
        #     test_V = np.expand_dims(test_V, axis=0)
        #     test_Y = np.expand_dims(test_Y, axis=0)
        #
        # def turnIntoSequence(t_x, t_y, repeat, length=10, predictNxt=1, overlap=4):
        #     x_arr = []
        #     y_arr = []
        #     for traj_id in range(repeat):
        #         offset = length - overlap
        #         for i in range(0, len(t_x[traj_id]) - length - predictNxt, offset):
        #             end_index = i + length + predictNxt
        #             x_arr.append(t_x[traj_id][i:end_index])
        #             y_arr.append(t_y[traj_id][end_index])
        #     return np.array(x_arr), np.array(y_arr)
        #
        # train_seq_U, train_seq_Y = turnIntoSequence(train_U, train_Y, len(train_U))
        # train_seq_V, _ = turnIntoSequence(train_V, train_Y, len(train_V))
        # train_seq_X2, _ = turnIntoSequence(train_Y, train_Y, len(train_Y))
        # test_seq_U, test_seq_Y = turnIntoSequence(test_U, test_Y, len(test_U))
        # test_seq_V, _ = turnIntoSequence(test_V, test_Y, len(test_V))
        # test_seq_X2, _ = turnIntoSequence(test_Y, test_Y, len(test_Y))
        #
        # X1_train = np.expand_dims(train_seq_U, axis=-1)
        # X2_train = np.expand_dims(train_seq_V, axis=-1)
        # X3_train = np.expand_dims(train_seq_X2, axis=-1)
        # y_train = np.expand_dims(train_seq_Y, axis=-1)
        # X1_test = np.expand_dims(test_seq_U, axis=-1)
        # X2_test = np.expand_dims(test_seq_V, axis=-1)
        # X3_test = np.expand_dims(test_seq_X2, axis=-1)
        # y_test = np.expand_dims(test_seq_Y, axis=-1)

        sequencer = LOMUQ_sequencer(batch_size=batch_size, image_size=mapsize, target_size=(32, 32), input_timesteps=n_steps, output_timesteps=10,
                                    dir_path=dir_path, save_to_dir="/media/christian/DATA/data/LOMUQ/out", save_format="png")
        sequencer.set_nsteps(epoch_steps)
        sequencer.prepareProcessedInput()


        class MonteCarloDropout(keras.layers.Dropout):
            def call(self, inputs):
                return super().call(inputs, training=True)

        # dshape = X1_train.shape
        dshape = (batch_size, 10, 32, 32, 1)
        samples, timesteps, rows, columns, features = dshape

        visible1 = Input(shape=(None, rows, columns, features))

        model1 = ConvLSTM2D(filters=64, kernel_size=(10, 10), activation='relu', padding="Same", return_sequences=True)(
            visible1)
        model1 = BatchNormalization()(model1)
        model1 = MonteCarloDropout(0.2)(model1)
        model1 = ConvLSTM2D(filters=64, kernel_size=(5, 5), activation='relu', padding="Same", return_sequences=False)(
            model1)
        model1 = BatchNormalization()(model1)
        model1 = MonteCarloDropout(0.2)(model1)
        # model1 = Dense(15,activation='relu')(model1)

        visible2 = Input(shape=(None, rows, columns, features))

        model2 = ConvLSTM2D(filters=64, kernel_size=(10, 10), activation='relu', padding="Same", return_sequences=True)(
            visible2)
        model2 = BatchNormalization()(model2)
        model2 = MonteCarloDropout(0.2)(model2)
        model2 = ConvLSTM2D(filters=64, kernel_size=(5, 5), activation='relu', padding="Same", return_sequences=False)(
            model2)
        model2 = BatchNormalization()(model2)
        model2 = MonteCarloDropout(0.2)(model2)

        visible3 = Input(shape=(None, rows, columns, features))

        model3 = ConvLSTM2D(filters=64, kernel_size=(10, 10), activation='relu', padding="Same", return_sequences=True)(
            visible3)
        model3 = BatchNormalization()(model3)
        model3 = MonteCarloDropout(0.2)(model3)
        model3 = ConvLSTM2D(filters=64, kernel_size=(5, 5), activation='relu', padding="Same", return_sequences=False)(
            model3)
        model3 = BatchNormalization()(model3)
        model3 = MonteCarloDropout(0.2)(model3)

        merge = concatenate([model1, model2, model3])

        dense = Dense(200, activation='relu')(merge)
        Output = Dense(1)(dense)

        model = Model(inputs=[visible1, visible2, visible3], outputs=Output)
        model.compile(optimizer='adam', loss='mae')
        model.summary()

        epochs = 30
        # batch_size = sequencer.batch_size
        nsteps = int(np.ceil(float(samples)/float(batch_size)))

        Early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        modelCheckpoint = ModelCheckpoint('/media/christian/DATA/data/LOMUQ/out/best_version1.hdf5', save_best_only=True)

        # Fit the model to the training data.
        # [X1_train, X2_train, X3_train], y_train,
        # ([X1_test, X2_test, X3_test], y_test)

        model_fitting_history = model.fit_generator(sequencer,
            steps_per_epoch=epoch_steps,
            epochs=epochs,
            validation_data=sequencer,
            # shuffle=True,
            callbacks=[Early_stopping, modelCheckpoint]
        )

        # model_fitting_history = model.fit(x=[X1_train, X2_train, X3_train],
        #     y=y_train,
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     validation_data=([X1_test, X2_test, X3_test], y_test),
        #     shuffle=True,
        #     callbacks=[Early_stopping, modelCheckpoint]
        # )

        model.save("/media/christian/DATA/data/LOMUQ/out/BestModel_sept.hdf5")
        with open(mhistfile.format(epoch=epochs), 'wb') as file:
            pickle.dump(model_fitting_history.history, file)

    except:
        with open("/media/christian/DATA/data/LOMUQ/out/exceptions1.log", "a") as logfile:
            traceback.print_exc(file=logfile)
        raise
