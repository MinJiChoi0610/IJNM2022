# <predict_resource_RNN.py>
#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb
# code is adopted from the above website
# use env tf
#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean




class RNN_resource_modeling:
    def __init__(self, trainfile,threshold_traffic, threshold_cpu,max_scaleup):
        tf.random.set_seed(0)
        self.max_scale=max_scaleup
        self.threshold_traffic=threshold_traffic
        self.threshold_cpu=threshold_cpu

        df = pd.read_csv(trainfile,
                         header=None, sep=',')
        df.columns = ['TIME', 'CPU', 'MEMORY', 'IO', 'IO', 'TRAFFIC', 'TRAFFIC-TX']
        self.target_names = ['TRAFFIC']

        shift_steps = 3  # Number of hours.
        df_targets = df[self.target_names].shift(-shift_steps)

        x_data = df.values[0:-shift_steps]
        y_data = df_targets.values[:-shift_steps]

        num_data = len(x_data)
        train_split = 0.9
        self.num_train = int(train_split * num_data)
        self.num_test = num_data - self.num_train
        x_train = x_data[0:self.num_train]
        x_test = x_data[self.num_train:]

        self.y_train = y_data[0:self.num_train]
        self.y_test = y_data[self.num_train:]


        self.num_x_signals = x_data.shape[1] # used


        self.num_y_signals = y_data.shape[1]


        self.x_scaler = MinMaxScaler()
        self.x_train_scaled = self.x_scaler.fit_transform(x_train)

        self.x_test_scaled = self.x_scaler.transform(x_test)

        self.y_scaler = MinMaxScaler()
        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train)
        self.y_test_scaled = self.y_scaler.transform(self.y_test)


        self.warmup_steps = 30
        self.model = Sequential()

    def batch_generator(self, batch_size, sequence_length):
        """
        Generator function for creating random batches of training-data.
        """
        # Infinite loop.
        while True:
            # Allocate a new array for the batch of input-signals.
            x_shape = (batch_size, sequence_length, self.num_x_signals)
            x_batch = np.zeros(shape=x_shape, dtype=np.float16)

            # Allocate a new array for the batch of output-signals.
            y_shape = (batch_size, sequence_length, self.num_y_signals)
            y_batch = np.zeros(shape=y_shape, dtype=np.float16)

            # Fill the batch with random sequences of data.
            for i in range(batch_size):
                # Get a random start-index.
                # This points somewhere into the training-data.
                idx = np.random.randint(self.num_train - sequence_length)

                # Copy the sequences of data starting at this index.
                x_batch[i] = self.x_train_scaled[idx:idx + sequence_length]
                y_batch[i] = self.y_train_scaled[idx:idx + sequence_length]


            yield (x_batch, y_batch)

    def train(self):
        #batch_size = 256
        batch_size = 70
        sequence_length = 35
        generator = self.batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)

        x_batch, y_batch = next(generator)

        validation_data = (np.expand_dims(self.x_test_scaled, axis=0),
                   np.expand_dims(self.y_test_scaled, axis=0))



        self.model.add(GRU(units=512,
              return_sequences=True,  input_shape=(None, self.num_x_signals,)))

        self.model.add(Dense(self.num_y_signals, activation='linear'))

        if False:
            from tensorflow.python.keras.initializers import RandomUniform

            # Maybe use lower init-ranges.
            init = RandomUniform(minval=-0.05, maxval=0.05)

            self.model.add(Dense(self.num_y_signals,
                            activation='linear',
                            kernel_initializer=init))



        optimizer = RMSprop(lr=1e-3)

        self.model.compile(loss=self.loss_mse_warmup, optimizer=optimizer)

        print(self.model.summary())

        path_checkpoint = '23_checkpoint.keras'
        callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_weights_only=True,
                                              save_best_only=True)

        callback_early_stopping = EarlyStopping(monitor='val_loss',
                                                patience=5, verbose=1)

        callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                           histogram_freq=0,
                                           write_graph=False)

        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.1,
                                               min_lr=1e-4,
                                               patience=0,
                                               verbose=1)

        callbacks = [callback_early_stopping,
                     callback_checkpoint,
                     callback_tensorboard,
                     callback_reduce_lr]

        directory = 'D:\POSTECH-paper-work\Prof-Yu-SFC-Implementaion\scaling\TensorFlow-Tutorials-d5f33973570fe6ef9c78c8a38c7449a932c81010\TensorFlow-Tutorials-d5f33973570fe6ef9c78c8a38c7449a932c81010\logs'

        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2),
            tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint),
            tf.keras.callbacks.TensorBoard(log_dir=os.path.abspath(directory))
        ]

        print("before model.fit")
        self.model.fit(x=generator,
                  epochs=10,
                  steps_per_epoch=10,
                  validation_data=validation_data)
        print("after model.fit")

        try:
            self.model.load_weights(path_checkpoint)
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)

        result = self.model.evaluate(x=np.expand_dims(self.x_test_scaled, axis=0),
                                y=np.expand_dims(self.y_test_scaled, axis=0))

        print("loss (test-set):", result)



    def loss_mse_warmup(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error between y_true and y_pred,
        but ignore the beginning "warmup" part of the sequences.

        y_true is the desired output.
        y_pred is the model's output.
        """

        # The shape of both input tensors are:
        # [batch_size, sequence_length, num_y_signals].

        # Ignore the "warmup" parts of the sequences
        # by taking slices of the tensors.
        y_true_slice = y_true[:, self.warmup_steps:, :]
        y_pred_slice = y_pred[:, self.warmup_steps:, :]

        # These sliced tensors both have this shape:
        # [batch_size, sequence_length - warmup_steps, num_y_signals]

        # Calculat the Mean Squared Error and use it as loss.
        mse = mean(square(y_true_slice - y_pred_slice))


        return mse

    def test(self, x_data):
        self.x_test_scaled = self.x_scaler.transform(x_data)
        x = self.x_test_scaled
        x = np.expand_dims(x, axis=0)
        y_pred = self.model.predict(x)
        y_pred_rescaled = self.y_scaler.inverse_transform(y_pred[0])

        signal_pred = y_pred_rescaled[:, 0]
        server_predicted = self.server_needed(signal_pred)
        return y_pred_rescaled, server_predicted


    def plot_comparison_with_scaling(self, start_idx, length=100, train=False):
        """
        Plot the predicted and true output-signals.
        :param start_idx: Start-index for the time-series.
        :param length: Sequence-length to process and plot.
        :param train: Boolean whether to use training- or test-set.
        """
        t=0
        if train:
            x = self.x_train_scaled
            y_true = self.y_train
        else:
            x = self.x_test_scaled
            y_true = self.y_test

        end_idx = start_idx + length

        x = x[start_idx:end_idx]
        y_true = y_true[start_idx:end_idx]

        # Input-signals for the model.
        x = np.expand_dims(x, axis=0)

        # Use the model to predict the output-signals.


        y_pred = self.model.predict(x) # model is predicting here

        # The output of the model is between 0 and 1.# Do an inverse map to get it back to the scale # of the original data-set.
        y_pred_rescaled = self.y_scaler.inverse_transform(y_pred[0])

        # For each output-signal.
        for signal in range(len(self.target_names)):
            signal_pred = y_pred_rescaled[:, signal]

            signal_true = y_true[:, signal]
            server_predicted = self.server_needed_list(signal_pred)

            server_needed = self.server_needed_list(signal_true)

            under_provision, over_provision = self.sla_violation(server_predicted, server_needed)

            print("Underprovisioning count :", under_provision)
            print("Overprovisioning count :", over_provision)

            under_provision_thresh, over_provision_thresh = self.sla_violation_threashold(server_needed)

            print("Underprovisioning count threshold based :", under_provision_thresh)
            print("Overprovisioning count threshold based:", over_provision_thresh)


            # Make the plotting-canvas bigger.
            plt.figure(figsize=(15, 5))


            # Plot and compare the two signals.
            t= np.arange(0, len(signal_pred)*5,5)
            plt.plot(t, signal_true, label='true', linewidth=4.0)
            plt.plot(t, signal_pred, label='pred', linewidth=4.0)

            # Plot grey box for warmup-period.
            p = plt.axvspan(0, self.warmup_steps, facecolor='black', alpha=0.15, linewidth=4.0)

            # Plot labels etc.
            plt.ylabel(self.target_names[signal] + "(Mbps)", fontsize=20)
            plt.xlabel('Time (m)',fontsize=20)
            plt.legend(fontsize=20)
            plt.show()

            plt.figure(figsize=(15, 5))
            plt.plot(t, server_needed, label='servers_needed')
            plt.plot(t, server_predicted, label='servers_pred')
            p = plt.axvspan(0, self.warmup_steps, facecolor='black', alpha=0.15)

            # Plot labels etc.
            plt.ylabel("#Servers" )
            plt.xlabel('Time (m)')
            plt.legend()
            plt.show()

    def sla_violation(self, server_predicted, server_needed):
        under_provision=0
        over_provision=0
        for i in range(len(server_predicted)):
            if server_predicted[i] == server_needed[i] :
                pass
            else:
                if server_predicted[i] < server_needed[i] :

                    under_provision=under_provision+1
                else:
                    if server_predicted[i] > server_needed[i] :
                        over_provision=over_provision+1
        return under_provision, over_provision

    def sla_violation_threashold(self, server_needed):
        under_provision=0
        over_provision=0
        for i in range(len(server_needed)-2):
            if server_needed[i] == server_needed[i+2] :
                pass
            else:
                if server_needed[i] < server_needed[i+2] :

                    under_provision=under_provision+1
                else:
                    if server_needed[i] > server_needed[i+2] :
                        over_provision=over_provision+1
        return under_provision, over_provision

    def server_needed_list(self, y_pred_rescaled):
        server_list=[]

        for i in range(len(y_pred_rescaled)):
            server_list.append(self.server_needed(y_pred_rescaled[i]))
        return server_list

    def server_needed(self, signal_pred):
        no_of_svr = 1
        threshold_request=self.threshold_traffic
        #i=2
        #for i in range(self.max_scale):
        if signal_pred >= threshold_request and signal_pred <= threshold_request*2:
            no_of_svr = 2
        else:
            if signal_pred >= threshold_request*2 and signal_pred <= threshold_request*3:
                no_of_svr = 3
            else:
                if signal_pred >= threshold_request*3 and signal_pred <= threshold_request*4:
                    no_of_svr = 4
                else:
                    if signal_pred >= threshold_request*4:
                        no_of_svr = 5
        return no_of_svr

if __name__ == '__main__':
    start1 = time.perf_counter()
    #RNN_model = RNN_resource_modeling('C:/Users/suman/request.txt') #CPU,Memory,I/O, Traffic in, Traffic Out for each VNF
    RNN_model = RNN_resource_modeling("measurement.csv", 2000, 40 , 5)  # CPU,Memory,I/O, Traffic in, Traffic Out for each VNF
    RNN_model.train()
    end1 = time.perf_counter()
    print("time required to train one week data :", end1 - start1)
    print("Plot comparision")
    RNN_model.plot_comparison_with_scaling(start_idx=0, length=1500, train=True)

    # Predict traffic demand.
    x_data = [[0,59.6386586,623099904,0,0,3239.364897,3238.365075]]
    print("Test for just one data value", x_data)
    predicted_traffic, predicted_servers = RNN_model.test(x_data)
    print("prediction for one data value",predicted_traffic[0][0],predicted_servers)
    print("threshold based for one data value", x_data[0][5], RNN_model.server_needed(x_data[0][5]))





#df['TRAFFIC'][200000:200000+1000].plot()
#df_org = weather.load_original_data()
#df_org.xs('Odense')['Temp']['2002-12-23':'2003-02-04'].plot();
