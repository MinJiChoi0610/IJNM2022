from __future__ import print_function
import ni_mon_client

from config import cfg
import ni_nfvo_client
import json
import datetime as dt
from dateutil.tz import gettz

class check_api :
    def __init__(self, prefix, sfc_name, waite_time):
        self.prefix=prefix
        self.sfc_name=sfc_name
        self.waite_time = waite_time
        self.resource_type = ["cpu_usage___value___gauge",
                         "memory_free___value___gauge",
                         "vda___disk_ops___read___derive",
                         "vda___disk_ops___write___derive",
                         "tap9f519bb4-61___if_packets___rx___derive",
                         "tap9f519bb4-61___if_packets___tx___derive"]
        # "if_packets___rx___derive", #"if_packets___tx___derive"
        self.vnf_id_name_pair = {
            "id": [],
            "name": []
        }

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
        print("VNF ID : ", self.vnf_id_name_pair["id"])
        print("VNF Name : ", self.vnf_id_name_pair["name"], "\n")

    def get_resource_utilization(self, sfc_name, start_time, end_time):
        # Get the SFC info in sfc_info variable
        nfvo_client_cfg = ni_nfvo_client.Configuration()
        nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
        ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))
        query = ni_nfvo_sfc_api.get_sfcs()
        sfc_info = [sfci for sfci in query if sfci.sfc_name == sfc_name]

        if len(sfc_info) == 0:
            return False

        sfc_info = sfc_info[-1]

        ni_mon_client_cfg = ni_mon_client.Configuration()
        ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
        api_instance = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

        no_of_tier = len(sfc_info.vnf_instance_ids)
        no_of_vnf = len(sfc_info.vnf_instance_ids[0])
        print("vnf_instance_ids : ", sfc_info.vnf_instance_ids)
        print("# of tiers : ", no_of_tier)
        print("# of VNFs : ", no_of_vnf , "\n")

        resource = []
        for type in self.resource_type:
            value = 0
            # print(no_of_vnf)

            for i in range(no_of_vnf):
                vnf = sfc_info.vnf_instance_ids[0][i]

                var = api_instance.get_measurement_types(vnf)

                query = api_instance.get_measurement(vnf, type, start_time, end_time)
                # print("query[0] :", query[0])
                # print("query[1] :", query[1])
                if query != []:
                    value = value + query[0].measurement_value
            resource.append(value)  # [CPU, Memory, Disk, Disk, Tx, Rx]

        print("Measurement Types : ", var, "\n")
        print("Resource : ", resource)
        print("Resource_cpu: ", resource[0])
        print("Resource_memory: ", resource[1])
        print("Resource_disk: ", resource[2])
        print("Resource_disk: ", resource[3])
        print("Resource_Traffic_TX: ", resource[4])
        print("Resource_Traffic_RX: ", resource[5])

        return resource



if __name__ == '__main__':

    with open('input.json') as f:
        sfc = json.load(f)
        prefix= sfc["prefix"]
        sfc_name = sfc["prefix"]+"_sfc"
        waite_time = sfc["wait_time"]


    # end_time = dt.datetime.now()
    # start_time = end_time - dt.timedelta(seconds=int(waite_time))
    end_time = dt.datetime.now(gettz('Asia/Seoul'))  # datetime | ending time to get the measurement
    start_time = end_time - dt.timedelta(seconds=int(waite_time))
    print("End_time :", end_time)
    check_api = check_api(prefix, sfc_name, int(waite_time))
    check_api.get_vnfs()
    check_api.get_resource_utilization(sfc_name, start_time, end_time)


