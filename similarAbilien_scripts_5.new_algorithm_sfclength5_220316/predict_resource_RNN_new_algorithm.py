# <predict_resource_RNN_new_algorithm.py>
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

import ni_mon_client
from config import cfg
import ni_nfvo_client
import json

class RNN_resource_modeling:
    def __init__(self, trainfile, threshold_cpu_upper, threshold_cpu_lower, threshold_traffic_upper, threshold_traffic_lower, max_scaleup):
        tf.random.set_seed(0)
        self.max_scale=max_scaleup
        self.threshold_cpu_upper = threshold_cpu_upper
        self.threshold_traffic_upper = threshold_traffic_upper

        df = pd.read_csv(trainfile, header=None, sep=',')
        df.columns = ['TIME', 'CPU', 'MEMORY', 'IO', 'IO', 'TRAFFIC-RX', 'TRAFFIC-TX']
        self.target_names = ['CPU', 'TRAFFIC-RX']

        shift_steps = 3  # Number of hours. Predicting 3minutes later traffic.
        df_targets = df[self.target_names].shift(-shift_steps)

        x_data = df.values[0:-shift_steps]
        #print("shape of x_data : df.values[0:-shift_steps] : ", x_data.shape)
        #measurement.csv 파일이 1500행일 때, x_data.shape = (1497, 7)
        y_data = df_targets.values[:-shift_steps]
        #print("shape of y_data : df_targets.values[:-shift_steps] : ", y_data.shape)
        #measurement.csv 파일이 1500행일 때, y_data.shape = (1497, 2) : 2 = self.target_names의 요소의 개수

        num_data = len(x_data)
        train_split = 0.9
        self.num_train = int(train_split * num_data) # 1497*0.9=1347
        self.num_test = num_data - self.num_train # 1497-1347=150
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


        self.warmup_steps = 50
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
                # idx = np.random.randint(self.num_train - sequence_length)

                # Copy the sequences of data starting at this index.
                x_batch[i] = self.x_train_scaled[idx:idx + sequence_length]
                y_batch[i] = self.y_train_scaled[idx:idx + sequence_length]


            yield (x_batch, y_batch)

    def train(self):
        #batch_size = 256
        batch_size =512
        sequence_length = 288
        generator = self.batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)

        x_batch, y_batch = next(generator)

        validation_data = (np.expand_dims(self.x_test_scaled, axis=0),
                   np.expand_dims(self.y_test_scaled, axis=0))



        self.model.add(GRU(units=512, return_sequences=True,  input_shape=(None, self.num_x_signals,)))

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

        #directory = 'D:\POSTECH-paper-work\Prof-Yu-SFC-Implementaion\scaling\TensorFlow-Tutorials-d5f33973570fe6ef9c78c8a38c7449a932c81010\TensorFlow-Tutorials-d5f33973570fe6ef9c78c8a38c7449a932c81010\logs'
        # directory = 'E:\2022.01\Scaling project_MinJi_2022.01\log'
        directory = 'E:\2022.01\Scaling project_MinJi_2022.01\Scripts_for_New_Algorithm_220214\log'
        # directory = '/home/ubuntu/ScalingProject2022/Scripts_for_tests/scripts_for_20220207_Wikki_RNN_Result/log'

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

    def test(self, x_data): # 여기서 받는 x_data는 average_resource
        self.x_test_scaled = self.x_scaler.transform(x_data) #평균0, 분산1로 범위 바꿈, x_test_scaled.shape=(150,7)
        x = self.x_test_scaled # 150*7
        x = np.expand_dims(x, axis=0) #첫번째 축에 차원 추가 x.shape = (1, 150, 7)
        y_pred = self.model.predict(x)
        y_pred_rescaled = self.y_scaler.inverse_transform(y_pred[0]) # 0~1사이로 normalize된 것을 원래대로 회복
        # y_pred_rescaled = self.y_scaler.inverse_transform(y_pred)

        cpu_pred = y_pred_rescaled[:, 0]
        traffic_pred = y_pred_rescaled[:, 1]
        Total_server_predicted = []

        Threshold = [self.threshold_cpu_upper, self.threshold_traffic_upper]

        for i in range(len(self.target_names)):
            server_predicted= self.server_needed_function(y_pred_rescaled[:, i], Threshold[i])
            Total_server_predicted.append(server_predicted)

            #return y_pred_rescaled, server_predicted
        return cpu_pred, traffic_pred, Total_server_predicted



    def plot_comparison_with_scaling(self, start_idx, length=100, train=False):
        """
        Plot the predicted and true output-signals.
        :param start_idx: Start-index for the time-series.
        :param length: Sequence-length to process and plot.
        :param train: Boolean whether to use training- or test-set.
        """
        t=0
        if train:
            x = self.x_train_scaled # x.shape = (1347, 7)
            y_true = self.y_train # y_true.shape = (1347, 2) : [[cpu, traffic], [cpu, traffic], ...]
        else:
            x = self.x_test_scaled # x.shape = (150, 7)
            y_true = self.y_test # y_true.shape = (150, 2)

        end_idx = start_idx + length

        x = x[start_idx:end_idx]
        # print("x.shape = : ", x.shape) # x.shape=(1347,7)
        y_true = y_true[start_idx:end_idx]
        # print("y_true.shape = : ", y_true.shape) # y_true.shape = (1347,2)

        # Input-signals for the model.
        x = np.expand_dims(x, axis=0)
        # print("x_expanded_dims.shape : ", x.shape) # x.shape=(1,1347,7)

        # Use the model to predict the output-signals.


        y_pred = self.model.predict(x)
        #print("y_pred : ", y_pred)
        # [[[ 0.06137885  0.08103882]
        #   [ 0.08498137  0.12346219]
        #   [ 0.09640671  0.13590807]
        #   ...
        #   [-0.006999   -0.06299549]
        #   [-0.00697549 -0.06286141]
        #   [-0.00717813 -0.0633306 ]]]

        # print("y_pred.shape : ", y_pred.shape) # y_pred.shape=(1,1347,2)

        # The output of the model is between 0 and 1.# Do an inverse map to get it back to the scale # of the original data-set.
        y_pred_rescaled = self.y_scaler.inverse_transform(y_pred[0])
        # print("y_pred_rescaled : ", y_pred_rescaled.shape) # y_pred_rescaled.shape = (1347,2)

        Total_server_predicted = []
        Total_server_needed = []

        Threshold = [self.threshold_cpu_upper, self.threshold_traffic_upper]

        # For each output-signal.
        for signal in range(len(self.target_names)): #self.target_names = ['CPU', 'TRAFFIC-RX']
            # signal=0 : cpu_pred, signal=1 : traffic_pred
            signal_pred = y_pred_rescaled[:, signal] # (1347,1)

            signal_true = y_true[:, signal] # y_true : y_train 또는 y_test
            server_predicted = self.server_needed_list(signal_pred, Threshold[signal]) #server_list 반환 : sever_predicted[cpu=0], server_predicted[traffic=1]
            Total_server_predicted.append(server_predicted) # Total_server_predicted = [[server_predicted list of cpu], [server_predicted list of traffic]]

            server_needed = self.server_needed_list(signal_true, Threshold[signal]) # server_list 반환 : sever_needed[cpu=0], server_needed[traffic=1]
            Total_server_needed.append(server_needed) # Total_server_predicted = [[server_needed list of cpu], [server_needed list of traffic]]

            under_provision, over_provision = self.sla_violation(server_predicted, server_needed)

            print("Underprovisioning count :", under_provision)
            print("Overprovisioning count :", over_provision)

            under_provision_thresh, over_provision_thresh = self.sla_violation_threshold(server_needed)

            print("Underprovisioning count threshold based :", under_provision_thresh)
            print("Overprovisioning count threshold based:", over_provision_thresh)


            # Make the plotting-canvas bigger.
            plt.figure(figsize=(15, 5))


            # Plot and compare the two signals.
            t = np.arange(0, len(signal_pred)*5, 5)
            plt.plot(t, signal_true, label='true', linewidth=4.0)
            plt.plot(t, signal_pred, label='pred', linewidth=4.0)

            # Plot grey box for warmup-period.
            p = plt.axvspan(0, self.warmup_steps, facecolor='black', alpha=0.15, linewidth=4.0)

            # Plot labels etc.
            if signal == 0 :
                plt.ylabel(self.target_names[signal] + "(%)", fontsize=20)
            if signal == 1 :
                plt.ylabel(self.target_names[signal] + "(Mbps)", fontsize=20)
            plt.xlabel('Time (m)',fontsize=20)
            plt.legend(fontsize=20)
            plt.show()

            plt.figure(figsize=(15, 5))

            if signal == 0 :
                plt.plot(t, server_needed, label='servers_needed_CPU')
                plt.plot(t, server_predicted, label='servers_pred_CPU')
            if signal == 1 :
                plt.plot(t, server_needed, label='servers_needed_Traffic')
                plt.plot(t, server_predicted, label='servers_pred_Traffic')

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

    def sla_violation_threshold(self, server_needed):
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

    def server_needed_list(self, y_pred_rescaled, Threshold): #y_pred_rescaled : (1347, 1)
        # y_pred_rescaled.shape = (1347,2)
        server_list=[]
        # Total_server_list = []

        for i in range(len(y_pred_rescaled)):
            server_list.append(self.server_needed_function(y_pred_rescaled[i], Threshold))
        # Total_server_list.append(server_list)

        return server_list

    def server_needed_function(self, signal_pred, threshold): # signal_pred = x_data[tier][resource]
        # 여기에서 cpu랑 memory 받아오기만 하자!
        # 여기 수정 필요 : 현재는 traffic만 받고 있음 : R함수를 여기서 구현...(self, signmal_pred, current # VNFs)
        no_of_svr = 1

        if (signal_pred >= threshold and signal_pred <= threshold*2) : # 기준 ~ 기준*2
            no_of_svr = 2
        else:
            if (signal_pred >= threshold*2 and signal_pred <= threshold*3) : # 기준*2 ~ 기준*3
                no_of_svr = 3
            else:
                if (signal_pred >= threshold*3 and signal_pred <= threshold*4) : # 기준*3 ~ 기준*4
                    no_of_svr = 4
                else:
                    if (signal_pred >= threshold*4) : # 기준*4 ~
                        no_of_svr = 5

        return no_of_svr

if __name__ == '__main__':

    with open('input.json') as f:
        sfc = json.load(f)
        prefix= sfc["prefix"]
        sfc_name = sfc["prefix"]+"_sfc" #Della-DC_sfc

    nfvo_client_cfg = ni_nfvo_client.Configuration()
    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))
    query = ni_nfvo_sfc_api.get_sfcs()
    sfc_info = [sfci for sfci in query if sfci.sfc_name == sfc_name]
    # print("sfc_info:",sfc_info)

    #if len(sfc_info) == 0:
        #return False

    sfc_info = sfc_info[-1]

    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    api_instance = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

    no_of_tier = len(sfc_info.vnf_instance_ids)

    start1 = time.perf_counter()
    #RNN_model = RNN_resource_modeling('C:/Users/suman/request.txt') #CPU,Memory,I/O, Traffic in, Traffic Out for each VNF
    threshold_cpu_upper = 20
    threshold_traffic_upper = 7500
    waite_time = 5
    RNN_model = RNN_resource_modeling("measurement.csv", threshold_cpu_upper, threshold_traffic_upper, waite_time)  # CPU,Memory,I/O, Traffic in, Traffic Out for each VNF
    RNN_model.train()
    end1 = time.perf_counter()
    print("time required to train one week data :", end1 - start1)
    print("Plot comparision")
    RNN_model.plot_comparison_with_scaling(start_idx=0, length=1500, train=True)

    '''
    # Predict traffic demand.
    x_data = [[0,59.6386586,623099904,0,0,3239.364897,3238.365075]]
    print("Test for just one data value", x_data)
    # predicted_traffic, predicted_servers = RNN_model.test(x_data)
    predicted_cpu, predicted_traffic, Total_server_predicted = RNN_model.test(x_data)


    print("prediction for one data value : ", "predicted_cpu : ", predicted_cpu[0][0], "predicted_traffic : ", predicted_traffic[0][0], "Total_server_predicted : ", Total_server_predicted + "\n")
    #print("threshold based for one data value", x_data[0][5], RNN_model.server_needed_function(x_data))
    print("threshold based for one data value : tier : ", "cpu : ", x_data[0][1], "traffic : ", x_data[0][5], "Server needed for CPU : ", RNN_model.server_needed_function(x_data[0][1], threshold_cpu_upper), "Server needed for Traffic : ", RNN_model.server_needed_function(x_data[0][5], threshold_traffic_upper) + "\n")

    for tiernum in range(0, no_of_tier):
        num_vnf_at_each_tier_c.append(len(sfc_info.vnf_instance_ids[tiernum]))  # C 정의 : 각 tier 별 VNF의 갯수
        Total_no_of_vnf = Total_no_of_vnf + len(sfc_info.vnf_instance_ids[tiernum])  # 전체 VNF 개수 파악 = Tier 별 VNF 갯수 더함
    print("# of VNFs at each tier : ", num_vnf_at_each_tier_c)
    print("Total # of VNFs : ", Total_no_of_vnf)
    '''



#df['TRAFFIC'][200000:200000+1000].plot()
#df_org = weather.load_original_data()
#df_org.xs('Odense')['Temp']['2002-12-23':'2003-02-04'].plot();
