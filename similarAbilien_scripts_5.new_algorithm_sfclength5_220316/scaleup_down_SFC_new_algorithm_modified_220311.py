# <scaleup_down_SFC.py>
from __future__ import print_function
import ni_mon_client
from config import cfg
import ni_nfvo_client
import time
import json
import datetime as dt
import os.path
import predict_resource_RNN_new_algorithm
from dateutil.tz import gettz

import numpy as np
import copy

class scaleup_down:
    def __init__(self, prefix, sfc_name, trainfile, scaling_duration, no_of_reading_per_unit_time, waite_time, threshold_cpu_upper, threshold_memory_upper, threshold_IO1_upper, threshold_IO2_upper, threshold_traffic_upper, threshold_traffic_TX_upper, threshold_cpu_lower, threshold_memory_lower, threshold_IO1_lower, threshold_IO2_lower, threshold_traffic_lower, threshold_traffic_TX_lower, max_scaleup):
        self.prefix=prefix
        self.sfc_name=sfc_name
        self.trainfile = trainfile
        self.scaling_duration = scaling_duration
        self.no_of_reading_per_unit_time = no_of_reading_per_unit_time
        self.waite_time = waite_time
        # Upper Threshold
        self.threshold_cpu_upper = threshold_cpu_upper
        self.threshold_memory_upper = threshold_memory_upper
        self.threshold_IO1_upper = threshold_IO1_upper
        self.threshold_IO2_upper = threshold_IO2_upper
        self.threshold_traffic_upper = threshold_traffic_upper
        self.threshold_traffic_TX_upper = threshold_traffic_TX_upper
        # Lower Threshold
        self.threshold_cpu_lower = threshold_cpu_lower
        self.threshold_memory_lower = threshold_memory_lower
        self.threshold_IO1_lower = threshold_IO1_lower
        self.threshold_IO2_lower = threshold_IO2_lower
        self.threshold_traffic_lower = threshold_traffic_lower
        self.threshold_traffic_TX_lower = threshold_traffic_TX_lower
        self.max_scaleup = max_scaleup
        self.resource_type = ["cpu_usage___value___gauge",
                         "memory_free___value___gauge",
                         "vda___disk_ops___read___derive",
                         "vda___disk_ops___write___derive",
                         "if_packets___rx___derive",
                         "if_packets___tx___derive"]
        self.vnf_id_name_pair = {
            "id": [],
            "name": []
        }
        self.log_scaling_file = open("log_scaling.csv", 'w')


    # log_scaling_event(): this function will log the current traffic, predicted traffic and predicted no of SFCs
    def log_scaling_event(self, sum_cpu, avg_cpu, predicted_cpu, sum_traffic, avg_traffic, predicted_traffic, scaled_vnf):
        # self.log_scaling_file.write(str(traffic) + ","+ str(predicted_traffic[0][0]) + "," + str(scaling_decision) + "\n")
        self.log_scaling_file.write(str(sum_cpu) + "," + str(avg_cpu) + "," + str(predicted_cpu[0][0])  + "," + str(sum_traffic) + ","+ str(avg_traffic) + ","+ str(predicted_traffic[0][0]) + "," + str(scaled_vnf))

    # update_sfc(sfc_info): Update SFC, main function to do auto-scaling
    # Input: updated sfc_info, which includes additional instances or removed instances
    # Output: none
    def update_sfc(self, sfc_info):
        sfc_update_spec = ni_nfvo_client.SfcUpdateSpec() # SfcUpdateSpec | Sfc Update info.
        sfc_update_spec.sfcr_ids = sfc_info.sfcr_ids
        sfc_update_spec.vnf_instance_ids = sfc_info.vnf_instance_ids

        nfvo_client_cfg = ni_nfvo_client.Configuration()
        nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
        ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

        ni_nfvo_sfc_api.update_sfc(sfc_info.id, sfc_update_spec)

    # find_type() : this function will find the type of the measurements
    def find_type(self, measurment_type,type):
        measurement_type_list=[]
        for i in range(len(measurment_type)):
            if measurment_type[i].find(type)!= -1:
                measurement_type_list.append(measurment_type[i])
        #print("measurement_type_list:", measurement_type_list)
        return measurement_type_list

    # get_vnf() : will store the VNF id and name pair filtered with the given prefix
    def get_vnfs(self):
        ni_mon_client_cfg = ni_mon_client.Configuration()
        ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
        api_instance = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

        vnf_list = api_instance.get_vnf_instances()

        for i in range(len(vnf_list)):
            if vnf_list[i].name.startswith(self.prefix):
                self.vnf_id_name_pair["id"].append(vnf_list[i].id)
                self.vnf_id_name_pair["name"].append(vnf_list[i].name)
        print("------------ Available VNFs for this SFC using prefix info----------")
        print(self.vnf_id_name_pair)

    # find_vnf_id(): this function will find the id of the VNF for given suffics for ex: if vnf name is della_DC_firewall_0_1. Fiven the _0_1, it will return id of this vnf
    def find_vnf_id(self, suffics):
        for i in range(len(self.vnf_id_name_pair["name"])):
            if self.vnf_id_name_pair["name"][i].endswith(suffics):
                return self.vnf_id_name_pair["id"][i]
        return 0

    # having_no_of_tier(sfc_name): Calculate the number of tier in the SFC.
    # Input:  sfc_name
    # Output: number of tiers in the SFC.
    def having_no_of_tier(self, sfc_name):
        # Get the SFC info in sfc_info variable
        nfvo_client_cfg = ni_nfvo_client.Configuration()
        nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
        ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))
        query = ni_nfvo_sfc_api.get_sfcs()
        sfc_info = [sfci for sfci in query if sfci.sfc_name == sfc_name]
        # print("sfc_info:",sfc_info)

        if len(sfc_info) == 0:
            return False

        sfc_info = sfc_info[-1]

        ni_mon_client_cfg = ni_mon_client.Configuration()
        ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
        api_instance = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

        no_of_tier = len(sfc_info.vnf_instance_ids)

        return no_of_tier

    # get_resource_utilization() : This function will return the resource CPU, memory, IO, Traffic in, Traffic out for given time duration for given sfc
    def get_resource_utilization(self, sfc_name, start_time, end_time):
        # Get the SFC info in sfc_info variable
        nfvo_client_cfg = ni_nfvo_client.Configuration()
        nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
        ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))
        query = ni_nfvo_sfc_api.get_sfcs()
        sfc_info = [sfci for sfci in query if sfci.sfc_name == sfc_name]
        #print("sfc_info:",sfc_info)

        if len(sfc_info) == 0:
            return False

        sfc_info = sfc_info[-1]

        ni_mon_client_cfg = ni_mon_client.Configuration()
        ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
        api_instance = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

        no_of_tier = len(sfc_info.vnf_instance_ids)
        #print("# of tier : ", no_of_tier)
        num_vnf_at_each_tier_c = [] # initialization
        Total_no_of_vnf = 0 #initialization
        # no_of_vnf = len(sfc_info.vnf_instance_ids[0])

        Total_resource = []
        Total_Avg_resource = []

        for tiernum in range(0, no_of_tier):
            resource = []
            Avg_resource = []
            num_vnf_at_each_tier_c.append(len(sfc_info.vnf_instance_ids[tiernum])) # C 정의 : 각 tier 별 VNF의 갯수
            #Total_no_of_vnf = Total_no_of_vnf + len(sfc_info.vnf_instance_ids[tiernum]) # 전체 VNF 개수 파악 = Tier 별 VNF 갯수 더함 #API call은 시간 소요됨.
            #print("# of VNFs at each tier : ", num_vnf_at_each_tier_c)  # C= 각 tier 별 VNF의 갯수 행렬로 print
            #print("Total # of VNFs : ", Total_no_of_vnf)

            for type in self.resource_type:
                value = 0
                for i in range(num_vnf_at_each_tier_c[tiernum]): # for i in range(no_of_vnf_in_tier1):
                    vnf = sfc_info.vnf_instance_ids[tiernum][i] # t=0 : firewall, t=1:ids, t=2:proxy ....  i=0 : 각 tier당 1번 VNF, i=1 : 각 tier당 2번 VNF, ...
                    # This was needed for traffic interface __rx__ and __tx__ as there are two type of traffic "control" and "data"
                    if type == "if_packets___rx___derive" or type == "if_packets___tx___derive" :
                    # if type == "if_packets___rx___derive" :
                        # var = api_instance.get_measurement_types(vnf)
                        # print(var)
                        type_list = self.find_type(api_instance.get_measurement_types(vnf), type)
                        for j in range(len(type_list)):
                            type_id = type_list[j]
                            # measurement_type_list[0]  + if_packets___rx___derive"
                            #end_time = dt.datetime.now()
                            #start_time = end_time - dt.timedelta(seconds=self.waite_time)
                            query = api_instance.get_measurement(vnf, type_id, start_time, end_time)
                            if query!=[]:
                                value = value + query[0].measurement_value
                                # avg_value = value / num_vnf_at_each_tier_c[t]

                    else: # for CPU, Memory, Disk
                        query = api_instance.get_measurement(vnf, type, start_time, end_time)
                        #print("query :", query)
                        if query!=[]:
                            value = value + query[0].measurement_value

                resource.append(value) # [CPU, Memory, Disk, Disk, Tx, Rx]
                # print("Current Tier :", tiernum, "resource for tier : ", resource, "\n")

            for l in range(len(self.resource_type)):
                Avg_resource.append(0) #여기 주석 처리해도 되는지 확인
                Avg_resource[l] = resource[l] / num_vnf_at_each_tier_c[tiernum]
            #print("Current Tier :", tiernum, "average resource for each tier : ", Avg_resource, "\n")

            Total_resource.append(list(resource))  # [[CPU, Memory, Disk, Disk, Tx, Rx],[CPU, Memory, Disk, Disk, Tx, Rx],[CPU, Memory, Disk, Disk, Tx, Rx]]
            #print("Total_resource : ", Total_resource, "\n")
            Total_Avg_resource.append(list(Avg_resource))  # 평균 [[CPU, Memory, Disk, Disk, Tx, Rx],[CPU, Memory, Disk, Disk, Tx, Rx],[CPU, Memory, Disk, Disk, Tx, Rx]]
            #print("Total_Avg_resource : ", Total_Avg_resource, "\n")

        #print("# of VNFs at each tier : ", num_vnf_at_each_tier_c)  # C= 각 tier 별 VNF의 갯수 행렬로 print
        #print("Total # of VNFs : ", Total_no_of_vnf)

        return Total_resource, Total_Avg_resource

    # store_measurement_to_train_RNN(self) : This function will measure the traffic and store it in measurment.csv file for traing the RNN model.
    def store_measurement_to_train_RNN(self):
        output_measurement = open(self.trainfile, 'w')
        no_of_tier = self.having_no_of_tier(self.sfc_name)
        for d in range(self.scaling_duration):
            #how many times you want to measure the resource
            max_resource=[]
            Total_max_resource = []
            for l in range(len(self.resource_type)):
                max_resource.append(0)

            for tiernum in range(no_of_tier):
                Total_max_resource.append(list(max_resource)) # Total_max_resource = [[tier1's resource 6개], [tier2's resource 6개], [tier3's resource 6개]]

            for t in range(self.no_of_reading_per_unit_time):
                #end_time = dt.datetime.now()
                end_time = dt.datetime.now(gettz('Asia/Seoul'))
                start_time = end_time - dt.timedelta(seconds=2)
                Total_resource, Total_Avg_resource =self.get_resource_utilization(self.sfc_name, start_time, end_time)
                print("Total_resource : ", Total_resource)
                '''
                print("Total_Avg_resource : ", Total_Avg_resource)


                for tiernum in range(no_of_tier):
                    for r in range(len(self.resource_type)):
                        if Total_max_resource[tiernum][r] < Total_Avg_resource[tiernum][r]:
                            Total_max_resource[tiernum][r] = Total_Avg_resource[tiernum][r]

            print("Total_max_resource:", Total_max_resource)
            max_resource_str = str(Total_max_resource)[1:len(str(Total_max_resource))-1]
            '''
            print("Total_resource : ", Total_resource)
            max_resource_str = str(Total_resource)[1:len(str(Total_resource))-1]
            output_measurement.write(str(d) + "," + max_resource_str)
            output_measurement.write("\n")

        output_measurement.close()

    # test_RNN_scaling(self) : This is main function with will train the RNN model with the existing traing file measurmenet.csv and then measure traffic untill "duration" and predict the traffic
    def test_RNN_scaling(self):
        RNN_model = predict_resource_RNN_new_algorithm.RNN_resource_modeling(self.trainfile, self.threshold_cpu_upper, self.threshold_cpu_lower, self.threshold_traffic_upper, self.threshold_traffic_lower,self.max_scaleup)  # CPU,Memory,I/O, Traffic in, Traffic Out for each VNF
        #print("sfc_name ;", self.sfc_name)
        start = time.perf_counter()
        RNN_model.train()
        end = time.perf_counter()
        print("time taken to train", start - end)
        RNN_model.plot_comparison_with_scaling(start_idx=0, length=1500, train=True)
        no_of_tier = self.having_no_of_tier(self.sfc_name)

        for d in range(self.scaling_duration):
            #how many times you want to measure the resource
            max_resource=[]
            for l in range(len(self.resource_type)): #[cpu, memory, diskread, diskwrite, rx, tx] = [0,0,0,0,0,0]
                max_resource.append(0) # initialize

            Total_max_resource_avg = []
            Total_max_resource_sum = []
            for tiernum in range(no_of_tier):
                #Total_max_resource_avg = [[tier1's resource 6개], [tier2's resource 6개], [tier3's resource 6개]]
                Total_max_resource_avg.append(list(max_resource))
                Total_max_resource_sum.append(list(max_resource))
            #print("Total_resource_format : ", Total_max_resource_avg)

            for t in range(self.no_of_reading_per_unit_time*3 - 6): # from 1 ~ 12 총 1 min(한 단계당 5초 * 12번 = 60초) # 12*3-6=30 * 5초 = 150초
                time1 = time.perf_counter()
                #end_time = dt.datetime.now()
                end_time = dt.datetime.now(gettz('Asia/Seoul'))
                start_time = end_time - dt.timedelta(seconds=self.waite_time)
                Total_resource, Total_Avg_resource = self.get_resource_utilization(self.sfc_name, start_time, end_time)
                #print("Total_resource : ", Total_resource)
                #print("Total_Avg_resource : ", Total_Avg_resource)

                # This is for smoothing.
                for tiernum in range(no_of_tier):
                    for r in range(len(self.resource_type)):
                        if Total_max_resource_avg[tiernum][r] < Total_Avg_resource[tiernum][r]:
                            Total_max_resource_avg[tiernum][r] = Total_Avg_resource[tiernum][r]
                        if Total_max_resource_sum[tiernum][r] < Total_resource[tiernum][r]:
                            Total_max_resource_sum[tiernum][r] = Total_resource[tiernum][r]

                time2 = time.perf_counter()
                sleep_if_needed = self.waite_time - (time2 - time1)
                #print("Sleep_if_needed, max_resource : \n", sleep_if_needed, max_resource)
                if sleep_if_needed > 0:
                    time.sleep(self.waite_time - (time2 - time1))

            print("Total_x_data_avg : ", d, Total_max_resource_avg)
            print("Total_x_data_sum : ", d, Total_max_resource_sum)

            # measurement is then sent to the RNN_model for prediction
            cpu_pred = []
            traffic_pred = []
            for tiernum in range(no_of_tier):
                cpu_pred.append(0)
                traffic_pred.append(0)

            for tiernum in range(no_of_tier): # CPU와 Traffic 량 예측 : cpu_pred = [tier1'cpu, tier2'cpu, tier3'cpu],
                cpu_pred[tiernum], traffic_pred[tiernum], Total_server_predicted[tiernum] = RNN_model.test(Total_max_resource_avg[tiernum])

                Total_max_resource_avg[tiernum][0] = cpu_pred[tiernum]
                Total_max_resource_avg[tiernum][4] = traffic_pred[tiernum]

            # based on the predicted traffic scaleup or scale down
            starttime = time.perf_counter()

            sfc_info, no_of_tier, num_vnf_at_each_tier_c, scaling_flag_for_each_tier = self.Calculate_ScaleInOut_Value(Total_max_resource_avg)
            scaled_VNF_at_each_tier = self.ActivateScaling(sfc_info, no_of_tier, num_vnf_at_each_tier_c, scaling_flag_for_each_tier)

            endtime = time.perf_counter()
            timetaken = endtime - starttime
            print("timetakenfordeployment :", endtime - starttime)
            d = d + int((timetaken % (self.no_of_reading_per_unit_time*self.waite_time)))


            for tiernum in range(no_of_tier):
                print("Tier : ", tiernum, "log: actual CPU usage, predicted CPU usage, scaling VNFs for CPU, actual traffic, predicted traffic, scaling VNFs for Traffic", Total_max_resource_avg[tiernum][0], Total_max_resource_avg[tiernum][4])
                # print("Scaling flag for each tier : ", scaling_flag_for_each_tier)
                self.log_scaling_event(Total_max_resource_sum[tiernum][0], Total_max_resource_avg[tiernum][0], " " , Total_max_resource_sum[tiernum][4], Total_max_resource_avg[tiernum][4], " " , scaled_VNF_at_each_tier[tiernum])
                self.log_scaling_file.write(",")
            self.log_scaling_file.write("\n")


            #print("max_resource:", Total_max_resource)
            print("---------------------------------------------------------------------------------------------------------------------------------")
            for tiernum in range(no_of_tier):
                #print("Total_max_resource[tiernum]", Total_max_resource[tiernum+1])
                #print("len(str(Total_max_resource[tiernum+1]))-1 :", len(str(Total_max_resource[tiernum+1])) - 1)
                max_resource_str = str(Total_max_resource_sum[tiernum])[1:len(str(Total_max_resource_sum[tiernum]))-1]
                output_measurement.write(max_resource_str + ",")
            output_measurement.write("\n")

        output_measurement.close()

    # test_threshold_scaling() : this function scaleup and down based on threshold values
    def test_threshold_scaling(self):
        output_measurement = open(self.trainfile, 'w')
        RNN_model = predict_resource_RNN_new_algorithm.RNN_resource_modeling("dummy.csv", self.threshold_cpu_upper, self.threshold_cpu_lower, self.threshold_traffic_upper, self.threshold_traffic_lower,self.max_scaleup)  # CPU,Memory,I/O, Traffic in, Traffic Out for each VNF
        #print("sfc_name ;", self.sfc_name)
        no_of_tier = self.having_no_of_tier(self.sfc_name)

        for d in range(self.scaling_duration): # scaling_duration = 1500
            #how many times you want to measure the resource
            max_resource=[]
            for l in range(len(self.resource_type)): #[cpu, memory, diskread, diskwrite, rx, tx] = [0,0,0,0,0,0]
                max_resource.append(0) # initialize

            Total_max_resource_avg = []
            Total_max_resource_sum = []
            for tiernum in range(no_of_tier):
                #Total_max_resource_avg = [[tier1's resource 6개], [tier2's resource 6개], [tier3's resource 6개]]
                Total_max_resource_avg.append(list(max_resource))
                Total_max_resource_sum.append(list(max_resource))
            #print("Total_resource_format : ", Total_max_resource_avg)

            for t in range(self.no_of_reading_per_unit_time*3 - 6): # from 1 ~ 12 총 1 min(한 단계당 5초 * 12번 = 60초) # 12*3-6=30 * 5초 = 150초
                time1 = time.perf_counter()
                #end_time = dt.datetime.now()
                end_time = dt.datetime.now(gettz('Asia/Seoul'))
                start_time = end_time - dt.timedelta(seconds=self.waite_time)
                Total_resource, Total_Avg_resource = self.get_resource_utilization(self.sfc_name, start_time, end_time)
                #print("Total_resource : ", Total_resource)
                #print("Total_Avg_resource : ", Total_Avg_resource)

                # This is for smoothing.
                for tiernum in range(no_of_tier):
                    for r in range(len(self.resource_type)):
                        if Total_max_resource_avg[tiernum][r] < Total_Avg_resource[tiernum][r]:
                            Total_max_resource_avg[tiernum][r] = Total_Avg_resource[tiernum][r]
                        if Total_max_resource_sum[tiernum][r] < Total_resource[tiernum][r]:
                            Total_max_resource_sum[tiernum][r] = Total_resource[tiernum][r]

                time2 = time.perf_counter()
                sleep_if_needed = self.waite_time - (time2 - time1)
                #print("Sleep_if_needed, max_resource : \n", sleep_if_needed, max_resource)
                if sleep_if_needed > 0:
                    time.sleep(self.waite_time - (time2 - time1))

            print("Total_x_data_avg : ", d, Total_max_resource_avg)
            print("Total_x_data_sum : ", d, Total_max_resource_sum)


            # based on the predicted traffic scaleup or scale down
            starttime = time.perf_counter()

            sfc_info, no_of_tier, num_vnf_at_each_tier_c, scaling_flag_for_each_tier = self.Calculate_ScaleInOut_Value(Total_max_resource_avg)
            scaled_VNF_at_each_tier = self.ActivateScaling(sfc_info, no_of_tier, num_vnf_at_each_tier_c, scaling_flag_for_each_tier)

            endtime = time.perf_counter()
            timetaken = endtime - starttime
            print("timetakenfordeployment :", endtime - starttime)
            d = d + int((timetaken % (self.no_of_reading_per_unit_time*self.waite_time)))


            for tiernum in range(no_of_tier):
                print("Tier : ", tiernum, "log: actual CPU usage, predicted CPU usage, scaling VNFs for CPU, actual traffic, predicted traffic, scaling VNFs for Traffic", Total_max_resource_avg[tiernum][0], Total_max_resource_avg[tiernum][4])
                # print("Scaling flag for each tier : ", scaling_flag_for_each_tier)
                self.log_scaling_event(Total_max_resource_sum[tiernum][0], Total_max_resource_avg[tiernum][0], " " , Total_max_resource_sum[tiernum][4], Total_max_resource_avg[tiernum][4], " " , scaled_VNF_at_each_tier[tiernum])
                self.log_scaling_file.write(",")
            self.log_scaling_file.write("\n")


            #print("max_resource:", Total_max_resource)
            print("---------------------------------------------------------------------------------------------------------------------------------")
            for tiernum in range(no_of_tier):
                #print("Total_max_resource[tiernum]", Total_max_resource[tiernum+1])
                #print("len(str(Total_max_resource[tiernum+1]))-1 :", len(str(Total_max_resource[tiernum+1])) - 1)
                max_resource_str = str(Total_max_resource_sum[tiernum])[1:len(str(Total_max_resource_sum[tiernum]))-1]
                output_measurement.write(max_resource_str + ",")
            output_measurement.write("\n")

        output_measurement.close()


    #sclaup_down() : this function will scale-up and scale-down the VNFs
    def Calculate_ScaleInOut_Value(self, Total_max_resource_avg):
        nfvo_client_cfg = ni_nfvo_client.Configuration()
        nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
        ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))
        query = ni_nfvo_sfc_api.get_sfcs()
        sfc_info = [sfci for sfci in query if sfci.sfc_name == self.sfc_name]

        if len(sfc_info) == 0:
            return False

        sfc_info = sfc_info[-1]

        ni_mon_client_cfg = ni_mon_client.Configuration()
        ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
        api_instance = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

        Total_no_of_vnf = 0  # initialization
        no_of_tier = len(sfc_info.vnf_instance_ids)
        # no_of_vnf = len(sfc_info.vnf_instance_ids[0])

        num_vnf_at_each_tier_c = []  # initialization
        scaling_flag_for_each_tier = []

        upper_Threshold = [self.threshold_cpu_upper, self.threshold_memory_upper, self.threshold_IO1_upper,
                           self.threshold_IO2_upper, self.threshold_traffic_upper, self.threshold_traffic_TX_upper]
        lower_Threshold = [self.threshold_cpu_lower, self.threshold_memory_lower, self.threshold_IO1_lower,
                           self.threshold_IO2_lower, self.threshold_traffic_lower, self.threshold_traffic_TX_lower]
        # print("upper Threshold :", upper_Threshold)
        # print("lower Threshold :", lower_Threshold)

        print("sfc_info.vnf_instance_ids: before Scaling up and down : ", sfc_info.vnf_instance_ids)

        scale_in_calculation = []
        scale_out_calculation = []
        for l in range(len(self.resource_type)):
            scale_in_calculation.append(0)  # initialize
            scale_out_calculation.append(0)  # initialize

        Total_scale_in_calculation = []
        Total_scale_out_calculation = []
        for tiernum in range(no_of_tier):
            Total_scale_in_calculation.append(list(scale_in_calculation))
            Total_scale_out_calculation.append(list(scale_out_calculation))

        # Calculation of R[n][k] : n = no_of_tier, k = # Resource Type
        R_Total_Avg_resource = copy.deepcopy(Total_max_resource_avg)

        multiplied_R_Total_Avg_resource = []
        for tiernum in range(no_of_tier):  # n
            multiplied_R_Total_Avg_resource.append(1) #[1, 1, 1]
            for k in range(len(self.resource_type)):  # k
                if k == 0 or k == 4:  # In case of CPU or Traffic
                    if Total_max_resource_avg[tiernum][k] > upper_Threshold[k]:
                        R_Total_Avg_resource[tiernum][k] = 0
                    elif Total_max_resource_avg[tiernum][k] < lower_Threshold[k]:
                        R_Total_Avg_resource[tiernum][k] = 1
                else: # Memory, I/O
                    R_Total_Avg_resource[tiernum][k] = 1
                multiplied_R_Total_Avg_resource[tiernum] = multiplied_R_Total_Avg_resource[tiernum] * R_Total_Avg_resource[tiernum][k]
        print("multiplied R in tiers : ", multiplied_R_Total_Avg_resource)

        for tiernum in range(no_of_tier):
            num_vnf_at_each_tier_c.append(len(sfc_info.vnf_instance_ids[tiernum]))
        print("num_vnf_at_each_tier_c : ", num_vnf_at_each_tier_c)

        for tiernum in range(no_of_tier):
            if multiplied_R_Total_Avg_resource[tiernum] == 0: # Scale Out!
                for type in range(len(self.resource_type)):  # resource_type = [CPU, Mem, IO, IO, Traffic-RX, TX]
                    if type == 0 or type == 4:  # In case of CPU or Traffic
                        Total_scale_out_calculation[tiernum][type] = int(int(Total_max_resource_avg[tiernum][type]) / int(upper_Threshold[type]) - num_vnf_at_each_tier_c[tiernum])
                        if Total_scale_out_calculation[tiernum][type] < 0:
                            Total_scale_out_calculation[tiernum][type] = 0
                    else:  # mem, IO, IO, TX
                        Total_scale_out_calculation[tiernum][type] = 0

                print("Tier #", tiernum, "Total_scale_out_calculation", Total_scale_out_calculation[tiernum])
                max_Total_scale_out = max(Total_scale_out_calculation[tiernum])
                print("Tier #", tiernum, "max_Total_scale_out_calculation", max_Total_scale_out)
                scaling_flag_for_each_tier.append(max_Total_scale_out)

            elif multiplied_R_Total_Avg_resource[tiernum] == 1 and num_vnf_at_each_tier_c[tiernum] > 1 : # Scale In!
                for type in range(len(self.resource_type)):  # resource_type = [CPU, Mem, IO, IO, Traffic-RX, TX]
                    if type == 0 or type == 4:  # In case of CPU or Traffic
                        Total_scale_in_calculation[tiernum][type] = int(num_vnf_at_each_tier_c[tiernum] - int(Total_max_resource_avg[tiernum][type]) / int(lower_Threshold[type]))
                        if Total_scale_in_calculation[tiernum][type] < 0:
                            Total_scale_in_calculation[tiernum][type] = 4

                    else:  # mem, IO, IO, TX
                        Total_scale_in_calculation[tiernum][type] = 4

                print("Tier #", tiernum, "Total_scale_in_calculation", Total_scale_in_calculation[tiernum])
                min_Total_scale_in = min(Total_scale_in_calculation[tiernum])
                print("Tier #", tiernum, "min_Total_scale_in_calculation", min_Total_scale_in)
                scaling_flag_for_each_tier.append(min_Total_scale_in * (-1))

            else:
                scaling_flag_for_each_tier.append(0)
            print("Tier :", tiernum, " Scaling up_down flag : ", scaling_flag_for_each_tier[tiernum])

        print("scaling flag for each tier : ", scaling_flag_for_each_tier)
        print("sfc_info.vnf_instance_ids: before Scaling up and down : ", sfc_info.vnf_instance_ids)

        return sfc_info, no_of_tier, num_vnf_at_each_tier_c, scaling_flag_for_each_tier

    def ActivateScaling(self, sfc_info, no_of_tier, num_vnf_at_each_tier_c, scaling_flag_for_each_tier):
        scaled_VNF_at_each_tier = []

        for tiernum in range(no_of_tier):
            if scaling_flag_for_each_tier[tiernum] > 0: #Scale Out
                for scaling in range(scaling_flag_for_each_tier[tiernum]):
                    suffics = str(tiernum) + "_" + str(scaling + num_vnf_at_each_tier_c[tiernum])
                    id = self.find_vnf_id(suffics)
                    if id != 0:
                        sfc_info.vnf_instance_ids[tiernum].append(id)
                    else:
                        # Create VNF instances code later
                        print("error no vnf exists for scaling")
                print("tierNum :", tiernum, "     scalingup ---- sfc_info.vnf_instance_ids:", sfc_info.vnf_instance_ids[tiernum])

            elif scaling_flag_for_each_tier[tiernum] < 0: #Scale In
                if num_vnf_at_each_tier_c[tiernum] + scaling_flag_for_each_tier[tiernum] > 0:
                    for scaling in range(abs(scaling_flag_for_each_tier[tiernum])):
                        # delete VNF instance code later
                        del sfc_info.vnf_instance_ids[tiernum][-1]
                    print("tierNum :", tiernum, "     scalingdown ---- sfc_info.vnf_instance_ids:", sfc_info.vnf_instance_ids[tiernum])
                else :
                    for scaling in range(abs(scaling_flag_for_each_tier[tiernum])):
                        # delete VNF instance code later
                        del sfc_info.vnf_instance_ids[tiernum][-1]
                    print("tierNum :", tiernum, "     scalingdown ---- sfc_info.vnf_instance_ids:", sfc_info.vnf_instance_ids[tiernum])

        print("sfc_info.vnf_instance_ids: after Scaling up and down : ", sfc_info.vnf_instance_ids)

        for tiernum in range(no_of_tier):
            scaled_VNF_at_each_tier.append(len(sfc_info.vnf_instance_ids[tiernum]))
        print("scaled VNF at each tier : ", scaled_VNF_at_each_tier)

        self.update_sfc(sfc_info)
        return scaled_VNF_at_each_tier


if __name__ == '__main__':

    with open('input.json') as f:
        sfc = json.load(f)
        prefix= sfc["prefix"]
        sfc_name = sfc["prefix"]+"_sfc" #Della-DC_sfc
        waite_time = sfc["wait_time"]
        no_of_reading_per_unit_time=int(sfc["-d_arg_wrk"]) / int(waite_time) # 60seconds/5 = 12seconds
        scaling_duration= sfc["scaling_duration"]  # 1500 , actual scaling duration will be 1500 * 10s
        # Upper Threshold
        threshold_cpu_util_upper = sfc["threshold_cpu_upper"]  # SLI indicators
        threshold_mem_upper = sfc["threshold_memory_upper"]  # SLI indicators
        threshold_io1_upper = sfc["threshold_IO1_upper"]  # SLI indicators
        threshold_io2_upper = sfc["threshold_IO2_upper"]  # SLI indicators
        threshold_request_per_sec_upper = sfc["threshold_traffic_upper"] # SLI indicators
        threshold_traffic_tx_upper = sfc["threshold_traffic_TX_upper"]  # SLI indicators
        # Lower Threshold
        threshold_cpu_util_lower = sfc["threshold_cpu_lower"]  # SLI indicators
        threshold_mem_lower = sfc["threshold_memory_lower"]  # SLI indicators
        threshold_io1_lower = sfc["threshold_IO1_lower"]  # SLI indicators
        threshold_io2_lower = sfc["threshold_IO2_lower"]  # SLI indicators
        threshold_request_per_sec_lower = sfc["threshold_traffic_lower"]  # SLI indicators
        threshold_traffic_tx_lower = sfc["threshold_traffic_TX_lower"]  # SLI indicators
        max_scaleup=sfc["max_scalup"]
        scaling_algo = sfc["scaling_algo"]

        upper_Threshold = [threshold_cpu_util_upper, threshold_mem_upper, threshold_io1_upper, threshold_io2_upper, threshold_request_per_sec_upper, threshold_traffic_tx_upper]
        lower_Threshold = [threshold_cpu_util_lower, threshold_mem_lower, threshold_io1_lower, threshold_io2_lower, threshold_request_per_sec_lower, threshold_traffic_tx_lower]

        # this is the function for calling the scale up-down constructor.
        scaleup_down = scaleup_down(prefix, sfc_name, "measurement.csv", int(scaling_duration), int(no_of_reading_per_unit_time), int(waite_time), int(threshold_cpu_util_upper), int(threshold_mem_upper), int(threshold_io1_upper), int(threshold_io2_upper), int(threshold_request_per_sec_upper), int(threshold_traffic_tx_upper), int(threshold_cpu_util_lower), int(threshold_mem_lower), int(threshold_io1_lower), int(threshold_io2_lower), int(threshold_request_per_sec_lower), int(threshold_traffic_tx_lower), int(max_scaleup))

        scaleup_down.get_vnfs()
        # scaleup_down.sclaup_down(1)

        if scaling_algo=="RNN_based":
            if os.path.isfile('measurement.csv'):

                scaleup_down.test_RNN_scaling()

            else:
                scaleup_down.store_measurement_to_train_RNN()
        else:
            print("i am here")
            scaleup_down.test_threshold_scaling()

        scaleup_down.log_scaling_file.close()
