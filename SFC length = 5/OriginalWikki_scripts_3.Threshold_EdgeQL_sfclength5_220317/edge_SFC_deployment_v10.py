from __future__ import print_function
import ni_mon_client
from config import cfg
from ni_mon_client.rest import ApiException
import ni_nfvo_client
from ni_nfvo_client.rest import ApiException
from pprint import pprint
import numpy as np
import ast
import time
import copy
import math
import sys
import json
#import matplotlib.pyplot as plt

# This flag is for future use of scaling up and down
SCALEUP=5


#This function will create topology and gather all the infromation from test bed.
#The network configurations are defined in "network.json" file and stored at the same location of this script.
#This function will also divide the network in edge, neighbour and datacenter, fill in flavours information, reshuffle the edge, calculate the resources left of each server
#all these information will be stored in the structure called topology

def get_topology_svr_vnf(topology):
    with open('network.json') as f:
        topology = json.load(f)

    # create an instance of the API class
    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    api_instance = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))


    reject_deployment= topology["load_these_servers"]
    print ("Overload servers: ",reject_deployment)

    with open('input.json') as f:
        sfc = json.load(f)
        #print(sfc['prefix'])
        #print(sfc['vnfs'])
        #print("sfc[client]", sfc["client"])

        src_id = ""
        dst_id = ""
        edge_seq = ""

        src_id = get_switch_id_from_name(sfc["client"])
        dst_id = get_switch_id_from_name(sfc["destination"])
        if (src_id == "" or dst_id==""):
            print("Client or Destination name is not found")
        else:
            print("Core Switch Id :", topology["core_switch_id_for_edge"])
            edge_seq = get_edge_switch_seq(topology["core_switch_id_for_edge"], src_id)
            print("edge_seq :", edge_seq)
            if (edge_seq == ""):
                print("edge_seq is not found")
            else:
                topology["vnf_prefix"] =sfc["prefix"]
                topology["source_client"] = src_id
                topology["destination_client"]=dst_id
                topology["src_ip_prefix"] =sfc["src_ip_prefix"]
                topology["dst_ip_prefix"] =sfc["dst_ip_prefix"]
                topology["edge_seq"] = edge_seq
                topology["vnfs"] = sfc["vnfs"]



    coreid = topology["core_switch_id_for_edge"] # this will be given as an input
    edge = [] # store the list of edges
    svr = [] #stores the list of servers at edges


    links = api_instance.get_links() # get all the links

    #this loop will get all the edges that are attached to the core switch
    for i in range(len(links)):
        node1 = api_instance.get_link(links[i].id).node1_id
        node2 = api_instance.get_link(links[i].id).node2_id
        if (node1 == coreid and api_instance.get_node(node2).type == "switch"):
            edge.append(node2)
            #print("found and edge",node2)
        else:
            if(node2 == coreid and api_instance.get_node(node1).type == "switch"):
                edge.append(node1)
                #print("found an edge:",node1)


    print("-----------edge------------")
    print("edge", edge)

    # this function will change edge sequence. (the second parameter in sfc-request.txt file in previous code), in the new code its the edge switch attached to the client.
    # changing edge sequence will make that edge a local network
    edge = change_edge_sequence(edge, edge_seq)

    print("-----------edge after changing seq------------")
    print("edge", edge)

    if len(edge) != topology["edge_no"]: # if configuration file provides the wrong edge_no then fix it here
        topology["edge_no"]=len(edge)

    # this loop will get all the compute nodes attached to the edges
    print("number of edge switch :", topology["edge_no"])
    print("number of servers on each switch :", topology["svr_no"])

    for i in range(len(edge)):
        nu=0
        for j in range(len(links)):
            node1 = api_instance.get_link(links[j].id).node1_id
            node2 = api_instance.get_link(links[j].id).node2_id
            if (node1 == edge[i] and api_instance.get_node(node2).type == "compute"):
                if nu < int(topology["svr_no"]): # maintain the symmetry in the network
                    svr.append(node2)
                    nu=nu+1
                else:
                    break
        if nu < int(topology["svr_no"]):
            for k in range(int(topology["svr_no"])-nu):
                svr.append("SYMMETRY") # if svr_no is less than actual svr_no from topology, then add extra server with "SYMMETRY" keywork to maintain the svr_no

    print("--------Local Servers------------")
    print("svr :", svr)

    # get all vnfs / get one vnf and obtain its -node_id, flavour_id / if node id is this node then cpu_occupied+get vnf_flavour.cpu
    all_vnfs = api_instance.get_vnf_instances()

    print("--------------Resources left on each servers-------------------")
    #print(all_vnfs)

    vnf_flav_at_svr = []
    for i in range(len(svr)):
        vnf_flav_at_svr.append([])


    #print("vnf_flav_at_svr",vnf_flav_at_svr)

    for i in range(len(all_vnfs)):
        for j in range(len(svr)):
            if(str(all_vnfs[i].status) =="ERROR"): # if some VNFs are in ERROR state, catch that exception
                pass
            else:
                if (all_vnfs[i].node_id == svr[j]):
                    vnf_flav_at_svr[j].append(all_vnfs[i].flavor_id)


    svr_states=[]
    cpu_no=0
    mem_no=0

    # This loop will caclulate the resources on all the servers.
    for i in range(len(svr)):
        cpu_used=0
        mem_used=0
        for k in range(len(vnf_flav_at_svr[i])):
            try:
                cpu_used=cpu_used+api_instance.get_vnf_flavor(vnf_flav_at_svr[i][k]).n_cores
                mem_used =mem_used + api_instance.get_vnf_flavor(vnf_flav_at_svr[i][k]).ram_mb
            except ni_mon_client.rest.ApiException: # if some VNFs are in ERROR state, catch that exception
                pass

        #print("CPU Mem used at ", svr[i], "is", cpu_used,mem_used)
        #if svr[i] == "SYMMETRY":
        #    print("Total CPU Mem at ", svr[i], "is", 0,0)
        #else:
        #    print("Total CPU Mem at ", svr[i], "is", api_instance.get_node(svr[i]).n_cores, api_instance.get_node(svr[i]).ram_mb )

        # This loop will avoid all the servers that are in "load_these_servers" in network.json
        do_not_deploy_here = False
        for s in range(len(reject_deployment)):
            if reject_deployment[s] == svr[i] :
                do_not_deploy_here=True
                #print("Do no deploy", svr[i],reject_deployment[s])

        if do_not_deploy_here or svr[i]=="SYMMETRY":
            cpu = 0
            mem = 0
        else:
            cpu = (api_instance.get_node(svr[i]).n_cores - cpu_used) - 3  # server needs 4 to 5 cpu for its functioning, so we will reserve 5 cpu for that.
            mem = (api_instance.get_node(svr[i]).ram_mb - mem_used)

        if cpu < 0:
            cpu=0

        print("CPU & Mem left at ", svr[i], "is", cpu,mem)

        svr_states.append([cpu,mem])

        # store the minimum cpu and mem of the edge here, this is used later for claculating reward
        if cpu_no==0:
            cpu_no=cpu
        else:
            if cpu < cpu_no:
                cpu_no=cpu

        if mem_no==0:
            mem_no=mem
        else:
            if mem < mem_no:
                mem_no=mem

    if cpu_no <= 0:
        cpu_no = 1

    if mem_no<=0:
       mem_no=1


    # follow the same procedure as above for all the DC servers.
    # Note : Future work to improve this code based on advanced DC
    dc_local_svr_no = int(topology["dc_edge_no"]) * int(topology["dc_tor_no"])* int(topology["dc_svr_no"])
    dc_svr = [topology["core_switch_id_for_DC"]] # later we improve this part
    dc_svr_states = []

    vnf_flav_at_dc_svr = []
    for i in range(len(dc_svr)):
        vnf_flav_at_dc_svr.append([])

    for i in range(len(all_vnfs)):
        for j in range(len(dc_svr)):
            if all_vnfs[i].node_id == dc_svr[j]:
                vnf_flav_at_dc_svr[j].append(all_vnfs[i].flavor_id)

    for i in range(len(dc_svr)):
        # Note: future work to improve this code. get the unoccupied n_cores and ram_mb only
        cpu_used=0
        mem_used=0
        for k in range(len(vnf_flav_at_dc_svr[i])):
            try:
                cpu_used=cpu_used+api_instance.get_vnf_flavor(vnf_flav_at_dc_svr[i][k]).n_cores
                mem_used = mem_used + api_instance.get_vnf_flavor(vnf_flav_at_dc_svr[i][k]).ram_mb
            except ni_mon_client.rest.ApiException:
                pass

        print("CPU Mem used at dc", dc_svr[i], "is", cpu_used,mem_used)
        print("Total CPU Mem at dc", dc_svr[i], "is", api_instance.get_node(dc_svr[i]).n_cores, api_instance.get_node(dc_svr[i]).ram_mb )
        cpu=(api_instance.get_node(dc_svr[i]).n_cores - cpu_used) - 4 # server needs 4 to 5 cpu for its functioning, so we will reserve 5 cpu for that.
        mem=(api_instance.get_node(dc_svr[i]).ram_mb - mem_used)
        if cpu < 0:
            cpu=0
        print("CPU Mem left at ", dc_svr[i], "is", cpu,mem)

        dc_svr_states.append([cpu,mem])


    # Get all the VNF flvour cpu and memory resrouces.
    vnf_type = np.array(
        [[api_instance.get_vnf_flavor(topology["vnf_flav"][0]).n_cores, api_instance.get_vnf_flavor(topology["vnf_flav"][0]).ram_mb],
         [api_instance.get_vnf_flavor(topology["vnf_flav"][1]).n_cores, api_instance.get_vnf_flavor(topology["vnf_flav"][1]).ram_mb],
         [api_instance.get_vnf_flavor(topology["vnf_flav"][2]).n_cores, api_instance.get_vnf_flavor(topology["vnf_flav"][2]).ram_mb],
         [api_instance.get_vnf_flavor(topology["vnf_flav"][3]).n_cores, api_instance.get_vnf_flavor(topology["vnf_flav"][3]).ram_mb],
         [api_instance.get_vnf_flavor(topology["vnf_flav"][4]).n_cores, api_instance.get_vnf_flavor(topology["vnf_flav"][4]).ram_mb],
         [api_instance.get_vnf_flavor(topology["vnf_flav"][5]).n_cores, api_instance.get_vnf_flavor(topology["vnf_flav"][5]).ram_mb],
         [api_instance.get_vnf_flavor(topology["vnf_flav"][6]).n_cores, api_instance.get_vnf_flavor(topology["vnf_flav"][6]).ram_mb],
         [api_instance.get_vnf_flavor(topology["vnf_flav"][7]).n_cores, api_instance.get_vnf_flavor(topology["vnf_flav"][7]).ram_mb]])



    #print("svr",svr)
    #print("svr_states",svr_states)
    #print("vnf_flav",vnf_flav)
    #print("vnf_type",vnf_type)

    topology["svr"]=svr
    topology["dc_svr"] = dc_svr
    topology["svr_states"]=svr_states
    topology["dc_svr_states"] = dc_svr_states
    topology["cpu_no"]=cpu_no
    topology["mem_no"]=mem_no
    topology["vnf_type"] = vnf_type

    return topology

# This function will get the edge node id, based on the client name and ip address information configured in input.json file.
def get_edge_node(edge_name, ip):
    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    api_instance = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))
    nodes = api_instance.get_nodes()
    id=""
    for i in range(len(nodes)):
        if (nodes[i].name == edge_name & nodes[i].ip == ip):
            id= nodes[i].id
            break
    return id

# This function will get the edge node id, based on the client name and ip address information configured in input.json file.
def get_edge_switch_seq(coreid, vnf_id):

    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    api_instance = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

    edge_seq_id = ""
    edge_seq=""
    if (api_instance.get_node(coreid)):
        edge=[]
        links= api_instance.get_links()
        for i in range(len(links)):
            node1 = api_instance.get_link(links[i].id).node1_id
            node2 = api_instance.get_link(links[i].id).node2_id
            if (node1 == coreid and api_instance.get_node(node2).type == "switch"):
                edge.append(node2)
                # print("found and edge",node2)
            else:
                if (node2 == coreid and api_instance.get_node(node1).type == "switch"):
                    edge.append(node1)

        #print(api_instance.get_vnf_instance(vnf_id))
        id = api_instance.get_vnf_instance(vnf_id).node_id

        for i in range(len(links)):
            if (links[i].node1_id==id ):
                if (links[i].node2_id in edge):
                    edge_seq_id=links[i].node2_id
                    break
            else :
                if (links[i].node2_id==id):
                    if (links[i].node1_id in edge):
                        edge_seq_id = links[i].node1_id
                        break

        edge_seq=edge.index(edge_seq_id)

    return edge_seq

    node = api_instance.get_node(id)

    return edge_seq_id

# get the switch id from the name of the VNF
def get_switch_id_from_name(name):
    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    api_instance = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))
    vnf_instances = api_instance.get_vnf_instances()
    id=""
    for i in range(len(vnf_instances)):
        #print(vnf_instances[i].name)
        if (vnf_instances[i].name == name):
            print("Match found")
            id = vnf_instances[i].id
            break
    return id

# change the edge sequence based on the client
def change_edge_sequence(edge,edge_seq_id):
    #file = open("./sfc_config.txt", 'r')
    #line = file.readline()
    #sfc_req = ast.literal_eval(line)  # Converting string to list
    #edge_seq_id = sfc_req[1]  # edge_seq(엣지 스위치 번호) 추출

    #swap the edge ids
    if edge_seq_id > len(edge) or edge_seq_id < 0:
        print("Edge Sequence Id provided is wrong")
    else:
        if(edge_seq_id==0):
            pass
        else:
            swappingid = edge[0]
            edge[0] = edge[edge_seq_id]
            edge[edge_seq_id]=swappingid
    return edge


def check_vnf_running_w_timeout(vnf_id, timeout):
    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

    tout = timeout
    while tout > 0:
        # get vnf_instance, allow not found because vnf may not ready yet
        try:
            vnf_instance = ni_mon_api.get_vnf_instance(vnf_id)
            if vnf_instance.status == 'ACTIVE':
                break
        except NimonApiException as e:
            assert e.status == 404

        time.sleep(1)
        tout = tout - 1

    vnf_instance = ni_mon_api.get_vnf_instance(vnf_id)
    if vnf_instance.status == 'ACTIVE':
        return True
    return False

'''------------------- End of SFC placement and deployment Program ------------------------ '''
# This function will take vnf_chain, svrlist, and s_a_history(which stores the vnf deploymnet information) as input
# Then deploy each vnf on the servers
# Then create a sfcr and finally create the sfc
#def deploy_sfc(s_a_history,vnf_flav,svr,dc_svr,svr_no,dc_svr_no,vnf_name,vnf_prefix): # we have only one datacenter so, dc_svr_no is not used yet
def deploy_sfc(s_a_history, topology):
    ni_nfvo_client_cfg = ni_nfvo_client.Configuration()
    ni_nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_vnf_api = ni_nfvo_client.VnfApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))

    #ni_mon_client_cfg = ni_mon_client.Configuration()
    #ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    #api_instance = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

    list1=[]
    list2=[]
    list2_scaling=[]
    #list2.append(['ecf636a6-e4bf-49ed-a898-9580e8d71504'])

    can_deploy=True
    for j in range(0, len(s_a_history) - 1):
        if s_a_history[j][1] == -1:  # One of the VNFS couldnt find an appropriate place for deployment
            can_deploy=False
            break
    if  can_deploy == True :
        print("s_a_history: [")



        for j in range(0, len(s_a_history)-1):
            tier_vnf=[]
            #print(topology["vnf_flav"][s_a_history[j][0]])
            #print("Check if flavour is right",api_instance.get_vnf_flavor(topology["vnf_flav"][s_a_history[j][0]]))

            #print(svr[s_a_history[j][1]])
            user_data= "#cloud-config\npassword: dpnm\nchpasswd: { expire: False }\nssh_pwauth: True\nmanage_etc_hosts: true\nruncmd:\n  - sysctl -w net.ipv4.ip_forward=1\n  - ifconfig ens3 mtu 1400\n  - ifconfig ens4 mtu 1400\n"
            try:
                if s_a_history[j][1] < len(topology["svr"]) : # if s_a_hisotry[j][1] (deployemnt server) is at the edge
                    #print(topology["svr"][s_a_history[j][1]])
                    #print("Check ifnode is right",api_instance.get_node(topology["svr"][s_a_history[j][1]]))
                    body = ni_nfvo_client.VnfSpec(flavor_id=topology["vnf_flav"][ s_a_history[j][0]],image_id=topology["vnf_images"][j], node_name=topology["svr"][ s_a_history[j][1] ],vnf_name=topology["vnf_prefix"]+"_"+topology["vnf_names"][j]+"_"+str(j),user_data=user_data)
                    #print("[ " + topology["vnf_prefix"]+"_" + topology["vnf_names"][j]+"_"+str(j) + " , " + topology["svr"][s_a_history[j][1]] + " ]")
                    print("[ " + topology["vnf_flav"][ s_a_history[j][0]] + " , " + topology["svr"][s_a_history[j][1]] + " ]")
                else:
                    #print(topology["dc_svr"][s_a_history[j][1] - len(topology["svr"])]) # get the index of the datacenter server
                    body = ni_nfvo_client.VnfSpec(flavor_id=topology["vnf_flav"][s_a_history[j][0]], image_id=topology["vnf_images"][j], node_name=topology["dc_svr"][s_a_history[j][1] - len(topology["svr"])], vnf_name=topology["vnf_prefix"]+"_"+topology["vnf_names"][j] + "_"+str(j),user_data=user_data)
                    #print("[ " + topology["vnf_prefix"]+"_" + topology["vnf_names"][j] + "_" + str(j) + " , " + topology["svr"][s_a_history[j][1]] + " ]")
                    print("[ " + topology["vnf_flav"][ s_a_history[j][0]] + " , " + topology["svr"][s_a_history[j][1]] + " ]")

                api_response1 = ni_nfvo_vnf_api.deploy_vnf(body)
                assert check_vnf_running_w_timeout(api_response1, timeout=120) == True

                list1.append(api_response1)
                list2.append([api_response1])
                tier_vnf.append(api_response1)

                #pprint(api_response1)

                if SCALEUP > 1 : # we need to reserve the vnfs for scaling up later when traffic increases
                    for k in range(SCALEUP-1):
                        if s_a_history[j][1] < len(topology["svr"]) :
                            body = ni_nfvo_client.VnfSpec(flavor_id=topology["vnf_flav"][ s_a_history[j][0]],image_id=topology["vnf_images"][j], node_name=topology["svr"][ s_a_history[j][1] ],vnf_name=topology["vnf_prefix"]+"_"+topology["vnf_names"][j]+"_"+str(j)+"_"+str(k+1),user_data=user_data)
                        else:
                            body = ni_nfvo_client.VnfSpec(flavor_id=topology["vnf_flav"][s_a_history[j][0]], image_id=topology["vnf_images"][j], node_name=topology["dc_svr"][s_a_history[j][1] - len(topology["svr"])], vnf_name=topology["vnf_prefix"]+"_"+topology["vnf_names"][j] + "_"+str(j)+"_"+str(k+1),user_data=user_data)
                        api_response1 = ni_nfvo_vnf_api.deploy_vnf(body)

                        assert check_vnf_running_w_timeout(api_response1, timeout=120) == True
                        pprint(api_response1)
                        tier_vnf.append(api_response1)
                    list2_scaling.append(tier_vnf) # this is used in the case of scaling.
                print("list2_scaling:", list2_scaling)
            except ApiException as e:
                print("VNF could not be deployed\n" % e)

        #list2.append(['b92a93cf-483a-4f67-afe6-79a04917071b '])

        print(", [8888, nan]]")



        ni_nfvo_sfcr_api = ni_nfvo_client.SfcrApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))

        #print(list1)
        if SCALEUP > 1:
            print("------------------- deploying VNFs --------------------")
            print(list2_scaling)
            print("------------- deploying SFCr-------------")
            nameofsfcr = topology["vnf_prefix"] + "_sfcr"
            nameofsfc = topology["vnf_prefix"] + "_sfc"
            sfcr_spec = ni_nfvo_client.SfcrSpec(name=nameofsfcr,
                                                source_client=topology["source_client"],
                                                src_ip_prefix=topology["src_ip_prefix"],
                                                dst_ip_prefix=topology["dst_ip_prefix"],
                                                nf_chain=list1
                                                )

            api_response_sfc = ni_nfvo_sfcr_api.add_sfcr(sfcr_spec)
            pprint(api_response_sfc)

            ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))
            sfc_spec = ni_nfvo_client.SfcSpec(sfc_name=nameofsfc,
                                              sfcr_ids=[api_response_sfc],
                                              vnf_instance_ids=list2_scaling
                                              )

            api_response_sfc = ni_nfvo_sfc_api.set_sfc(sfc_spec)
            print("-----------deploying SFC-------------")
            pprint(api_response_sfc)
        else:
            print("------------------- deploying VNFs --------------------")
            print(list2_scaling)
            print("------------- deploying SFCr-------------")
            nameofsfcr=topology["vnf_prefix"]+"_sfcr"
            nameofsfc=topology["vnf_prefix"]+"_sfc"
            sfcr_spec = ni_nfvo_client.SfcrSpec(name=nameofsfcr,
                                            source_client=topology["source_client"],
                                            src_ip_prefix=topology["src_ip_prefix"],
                                            dst_ip_prefix=topology["dst_ip_prefix"],
                                            nf_chain=list1
                                            )

            api_response_sfc = ni_nfvo_sfcr_api.add_sfcr(sfcr_spec)
            pprint(api_response_sfc)

            ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))
            sfc_spec = ni_nfvo_client.SfcSpec(sfc_name = nameofsfc,
                                          sfcr_ids = [api_response_sfc],
                                          vnf_instance_ids = list2
                                          )

            api_response_sfc = ni_nfvo_sfc_api.set_sfc(sfc_spec)
            print("-----------deploying SFC-------------")
            pprint(api_response_sfc)
    else:
        print ("One of the VNF can not be deployed due to resource limitations on server")
    return list2_scaling


'''---------------------------------------------------------------------------------'''
# 정책 파라미터 theta_0을 무작위 행동 정책 pi로 변환하는 함수
def simple_convert_into_pi_from_theta(theta):
    '''단순 비율 계산'''

    [m, n] = theta.shape  # theta의 행렬 크기를 구함
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])  # 비율 계산
        #print(pi[i, :])
    pi = np.nan_to_num(pi)  # nan을 0으로 변환
    #print("pi",pi)
    return pi


'''------------------------------------------------------------------------'''
# Q러닝 알고리즘으로 행동가치 함수 Q를 수정---- a_next를 사용하지 않는다.

def Q_learning(s, a, r, s_next, Q, eta, gamma):
    if s_next == 8888:  # 목표 지점에 도달한 경우
        #print("End of a SFC deployment")
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
        #print("Q_1=", Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next,: ]) - Q[s, a])
        #print("Q_2=", Q[s, a])

    return Q

''' ----  np.random.choice()에서 종종 발생하는 에러(probabilities do not sum to 1) 대책 '''
def kahansum(input):
    summ = c = 0
    for num in input:
        y = num - c
        t = summ + y
        c = (t - summ) - y
        summ = t
    if summ == 0 :
        summ =1
    return summ


def svr_loc(edge_seq_id, p_act, action,topology):
    '''---- 참조정보----------------------------'''
    total_svr_list = np.arange(topology["total_svr_no"], dtype=int)
    local_svr_list = np.arange(topology["local_svr_no"], dtype=int)
    # print("전체 서버 번호=", total_svr_list)
    # print("로컬  서버 번호=", local_svr_list)

    self_svr_list = np.arange(topology["local_svr_no"] // topology["edge_no"] * (edge_seq_id - 1), topology["local_svr_no"] // topology["edge_no"] * edge_seq_id,
                              dtype='i')
    # print("자국 서버 번호=", self_svr_list)
    # print("dtype=",self_svr_list.dtype)

    neigh_svr_list = np.delete(local_svr_list, self_svr_list)
    # print(" 인접국 서버 번호=", neigh_svr_list)
    dc_svr_list = np.arange(topology["local_svr_no"], topology["total_svr_no"])
    # print(" DC 서버 번호=", dc_svr_list)
    '''--------------------------------------------'''

    self = self_svr_list.reshape(int(topology["tor_no"]), int(topology["svr_no"]))
    neigh = neigh_svr_list.reshape(int(topology["edge_no"]) - 1, int(topology["tor_no"]),
                                   int(topology["svr_no"]))  # 3차원으로 수정, neigh는 전체 스위치(edge_no) 수에서 self가 사용하는 것을 제외하므로
    dc = dc_svr_list.reshape(int(topology["dc_edge_no"]), int(topology["dc_tor_no"]), int(topology["dc_svr_no"]))
    # print("self=", self)
    # print("neigh=", neigh)
    # print("dc=", dc)

    # print("p_act",p_act)
    p_self_ret = np.where(self == p_act)
    #print("p_self_ret=", p_self_ret)
    #print("p_self_ret=", p_self_ret[0], p_self_ret[1])

    lc_p_self_ret = list(p_self_ret)  # list conversion, np.where에서 return된 tuple을 list로 변경하여 빈 값을 -1로 변경해줌(오류 방지 용).

    '''  # 출력 예: p_self_ret= (array([1], dtype=int64), array([0], dtype=int64)) 또는 p_self_ret= (array([], dtype=int64), array([], dtype=int64))'''

    if len(lc_p_self_ret[0]) == 0 and len(lc_p_self_ret[1]) == 0:  # 출력 예에서 값이 값이 없는 경우 연산을 위해 -1을 추가함(후자)
        lc_p_self_ret[0] = -1
        lc_p_self_ret[1] = -1
    #print("lc_p_self_ret=", tuple(lc_p_self_ret))
    #print("lc_p_self_ret=",lc_p_self_ret[0],lc_p_self_ret[1])

    p_neigh_ret = np.where(neigh == p_act)
    #print("p_neigh_ret=", p_neigh_ret)
    #print("p_neigh_ret=", p_neigh_ret[0], p_neigh_ret[1], p_neigh_ret[2])
    lc_p_neigh_ret = list(p_neigh_ret)  # list conversion
    if len(lc_p_neigh_ret[0]) == 0 and len(lc_p_neigh_ret[1]) == 0 and len(lc_p_neigh_ret[2]) == 0:
        lc_p_neigh_ret[0] = -1
        lc_p_neigh_ret[1] = -1
        lc_p_neigh_ret[2] = -1
    #print("lc_p_neigh_ret=", tuple(lc_p_neigh_ret))
    #print("lc_p_neigh_ret=",lc_p_neigh_ret[0], lc_p_neigh_ret[1], lc_p_neigh_ret[2])

    p_dc_ret = np.where(dc == p_act)
    lc_p_dc_ret = list(p_dc_ret)  # list conversion
    if len(lc_p_dc_ret[0]) == 0 and len(lc_p_dc_ret[1]) == 0 and len(lc_p_dc_ret[2]) == 0:
        lc_p_dc_ret[0] = -1
        lc_p_dc_ret[1] = -1
        lc_p_dc_ret[2] = -1

    self_ret = np.where(self == action)  # 자국에 대해 검사
    # print("self_ret=", self_ret)
    # print("self_ret=", self_ret[0], self_ret[1])        # [0]는 2차원 원소 번호, [1]은 1차원 원소 번호
    lc_self_ret = list(self_ret)  # list conversion
    if len(lc_self_ret[0]) == 0 and len(lc_self_ret[1]) == 0:
        lc_self_ret[0] = -1
        lc_self_ret[1] = -1
    # print("lc_self_ret=", tuple(lc_self_ret))
    # print("lc_self_ret=",lc_self_ret[0],lc_self_ret[1])

    neigh_ret = np.where(neigh == action)  # 인접국에 대해 검사
    # print("neigh_ret=", neigh_ret[0], neigh_ret[1], neigh_ret[2])
    # print("neigh_ret=",neigh_ret[0], neigh_ret[1])
    lc_neigh_ret = list(neigh_ret)
    if len(lc_neigh_ret[0]) == 0 and len(lc_neigh_ret[1]) == 0 and len(lc_neigh_ret[2]) == 0:
        lc_neigh_ret[0] = -1
        lc_neigh_ret[1] = -1
        lc_neigh_ret[2] = -1
    # print("lc_neigh_ret=", tuple(lc_neigh_ret))
    # print("lc_neigh_ret=", lc_neigh_ret[0], lc_neigh_ret[1], lc_neigh_ret[2])

    dc_ret = np.where(dc == action)  # DC
    lc_dc_ret = list(dc_ret)
    if len(lc_dc_ret[0]) == 0 and len(lc_dc_ret[1]) == 0 and len(lc_dc_ret[2]) == 0:
        lc_dc_ret[0] = -1
        lc_dc_ret[1] = -1
        lc_dc_ret[2] = -1

    return lc_p_self_ret, lc_p_neigh_ret, lc_p_dc_ret, lc_self_ret, lc_neigh_ret, lc_dc_ret


# VNF를 순차 처리하고 처리가 끝나면 빠져나오는 함수, 상태와 행동의 히스토리를 출력하도록 수정됨.
''' Numpy 배열은 immutable 이며, 항목을 변경 가능하지만 삭제할 수는 없다. 따라서 일반 배열로 sfc_req 배열을 정의 하자 '''
''' This function is called for each vnf in SFC, and it is main function for allocting space in local, neighbour or datacenter servers.'''

def allocate(Q,aa,sfc_req,svr_states_clone,epsilon,local_svr_list,self_svr_list,neigh_svr_list,dc_svr_list, s_a_history,L,r, theta_0,topology):
    vnf_type_no = sfc_req[-1][aa]  # sfc_req에서 vnf 번호를 포함한 single list를 추출하고 순서대로 대입,
    [cpu, mem] = topology["vnf_type"][vnf_type_no] * SCALEUP  # vnf_type[번호]별로 자원 요구량 추출

    # ---------------------------무작위 action의 선택 -------------------------------------------------#
    ''' pi_0로부터 vnf_type 번호와 자국, 인접국, DC 서버 범위에 해당하는 확률 행렬을 pi_1, pi_3 및 pi_5에 카피한다'''

    pi_0 = simple_convert_into_pi_from_theta(theta_0)

    pi_1 = pi_0[vnf_type_no, self_svr_list[0]:self_svr_list[-1] + 1].copy()

    # print("pi_1", pi_1)
    pi_svr_to_delete = np.append(self_svr_list, dc_svr_list)  # pi_3를 구할 때 사용
    pi_3 = np.delete(pi_0[vnf_type_no], pi_svr_to_delete)
    pi_5 = np.delete(pi_0[vnf_type_no], local_svr_list)  # DC에 대한 확률
    #print("pi_1", pi_1)
    #print("pi_3", pi_3)
    #print("pi_5", pi_5)
    pi_2 = np.zeros(len(self_svr_list))

    x = len(self_svr_list)
    for i in range(0, x):
        # pi_2[i]= pi_1[i]/np.sum(pi_1[:])    # 0으로 나누는 경우 경고: RuntimeWarning: invalid value encountered in double_scalars
        pi_2[i] = np.nan_to_num(pi_1[i] / kahansum(pi_1[:]))
    #print("pi_2",pi_2)


    pi_4 = np.zeros(len(neigh_svr_list))
    q = len(neigh_svr_list)
    for i in range(0, q):
        # pi_4[i]= pi_3[i]/np.sum(pi_3[:])    # 0으로 나누는 경우 경고: RuntimeWarning: invalid value encountered in double_scalars
        pi_4[i] = pi_3[i] / kahansum(pi_3[:])

    pi_6 = np.zeros(len(dc_svr_list))
    n = len(dc_svr_list)
    for i in range(0, n):
        # pi_6[i]= pi_5[i]/np.sum(pi_5[:])    # random.choice 사용 시 확률의 합은 1.0이어야 함 && 0으로 나누는 경우 경고를 피하기 위해서
        pi_6[i] = pi_5[i] / kahansum(pi_5[:])

    #print("pi_2", pi_2)
    #print("pi_4", pi_4)
    #print("pi_6", pi_6)

    # print("자국 서버에 설치 시도")
    ''' pi_1 행렬의 확률의 합이 1.0 이되도록 변환하여 그 결과를 pi_2에 저장한다. '''
    #print("np.sum(pi_1[:]) != 0:", np.sum(pi_2[:]))
    action = -1
    count=0
    svr_to_delete = []
    if (np.sum(pi_2[:]))!= 0:
        while (np.sum(pi_2[:])) != 0: # this loop will make sure that all the local servers are occupied before moving to neighbour servers
            count=count+1
            #print("count " ,count)
            if np.random.rand() < epsilon:
                #print("pi_2",pi_2)
                action = np.random.choice(self_svr_list, p=pi_2[:])  # 확률 ε로 무작위 행동을 선택함
                #print("Exploring count, vnf, action, cpu, mem left", count, aa, action,vnf_type_no,svr_states_clone[action][0] - [cpu],svr_states_clone[action][1] - [mem])
            else:
                svr_to_deploy = np.where(np.array(pi_2) != 0)
                #print("svr_to_deploy",svr_to_deploy[0])
                action = svr_to_deploy[0][np.nanargmax(Q[sfc_req[-1][aa], svr_to_deploy][0])]
                #print("Q[sfc_req[-1][aa], svr_to_deploy][0]",Q[sfc_req[-1][aa], svr_to_deploy][0])
                ''' 선택된 action에 따라 하나의 서버를 선택하고 설치 시도한다'''
                #print("Exploiting count, vnf, action, cpu, mem left", count, aa, action,vnf_type_no,svr_states_clone[action][0] - [cpu],svr_states_clone[action][
                #1] - [mem])

            if svr_states_clone[action][0] - [cpu] >= 0 and svr_states_clone[action][
                1] - [mem] >= 0:  # action은 선택된 서버 번호+1
                #print("svr_states_clone[action][0]", svr_states_clone[action][0])
                svr_states_clone[action][0] = svr_states_clone[action][0] - [cpu]
                svr_states_clone[action][1] = svr_states_clone[action][1] - [mem]
                #print("svr_states_clone[action][0]", svr_states_clone[action][0])
                #print("I am here and breaking the loop", count)

                if svr_states_clone[action][0] - [cpu] <= 0 or svr_states_clone[action][1] - [mem] <= 0:
                    # 특정 서버의 자원이 VNF_type 설치에 충분하면 값을 1로
                    theta_0[vnf_type_no][action] = 0
                    pi_0 = simple_convert_into_pi_from_theta(theta_0)
                    pi_1 = pi_0[vnf_type_no, self_svr_list[0]:self_svr_list[-1] + 1].copy()
                    for i in range(0, x):
                    # pi_2[i]= pi_1[i]/np.sum(pi_1[:])    # 0으로 나누는 경우 경고: RuntimeWarning: invalid value encountered in double_scalars
                        pi_2[i] = pi_1[i] / kahansum(pi_1[:])

                #print("action, cpu left, mem left", action, svr_states_clone[action][0] - [cpu],  svr_states_clone[action][1] - [mem])
                break
                # r=7    #########################
            else:
                ''' policy updation '''
                #print("theta_0[vnf_type_no][action]",theta_0[vnf_type_no][action],svr_states_clone[action][0],svr_states_clone[action][1])
                theta_0[vnf_type_no][action] = 0
                #print("theta_0[vnf_type_no][action]", theta_0[vnf_type_no][action])
                pi_0 = simple_convert_into_pi_from_theta(theta_0)
                pi_1 = pi_0[vnf_type_no, self_svr_list[0]:self_svr_list[-1] + 1].copy()
                for i in range(0, x):
                    # pi_2[i]= pi_1[i]/np.sum(pi_1[:])    # 0으로 나누는 경우 경고: RuntimeWarning: invalid value encountered in double_scalars
                    pi_2[i] = pi_1[i] / kahansum(pi_1[:])
                action = -1
                #svr_states_clone, s_a_history,L,r, action= allocate(aa,sfc_req,svr_states_clone, epsilon, local_svr_list, self_svr_list, neigh_svr_list, dc_svr_list, s_a_history, L, r)

    if action == -1 :  # this condition is introduced to avoid several if conditions, action = -1 means the previous step couldnt allocate the server
        count=0
        #print("2 Local server cannot be selected-> Attempt to select an adjacent server")
        # ---------------------------------------------------------------------------------------
        ''' pi_1 행렬의 확률의 합이 1.0 이되도록 변환하여 그 결과를 pi_2에 저장한다. '''

        if (np.sum(pi_4[:])) != 0:
            #while (np.sum(pi_4[:])) != 0:   # this loop will make sure that all the neighbour servers are occupied before moving to datacenter servers
            while (np.sum(pi_4[:])) != 0:
                count =count+1
                print("count ",count)
                if np.random.rand() < epsilon:
                    action = np.random.choice(neigh_svr_list, p=pi_4[:])  # 확률 ε로 무작위 행동을 선택함
                else:
                    ''' index=self_svr_list 및 dc_svr_list에 해당하는 원소를 Q에서 제외하고 '''
                    svr_to_delete = np.append(self_svr_list, dc_svr_list, axis=0)
                    #print("(np.where(np.array(pi_4) == 0))[0]",(np.where(np.array(pi_4) == 0))[0])
                    #print("(np.where(np.array(pi_4) == 0))[0]", neigh_svr_list[(np.where(np.array(pi_4) == 0))[0]])
                    svr_to_delete = np.append(svr_to_delete,neigh_svr_list[(np.where(np.array(pi_4) == 0))[0]], axis=0)
                    #print("svr_to_delete", svr_to_delete)
                    #print("np.delete(Q[sfc_req[-1][aa], :], svr_to_delete)",np.delete(Q[sfc_req[-1][aa], :], svr_to_delete))
                    neigh_svr_list1= neigh_svr_list[(np.where(np.array(pi_4) != 0))[0]]
                    action = neigh_svr_list1[np.nanargmax(np.delete(Q[sfc_req[-1][aa], :], svr_to_delete))]
                #print("neigh_action=", action)
                ''' 선택된 action에 따라 하나의 인접국 서버를 선택하고 설치 시도한다'''
                if svr_states_clone[action][0] - cpu >= 0 and svr_states_clone[action][
                    1] - mem >= 0:  # action은 선택된 서버 번호+1
                    svr_states_clone[action][0] = svr_states_clone[action][0] - [cpu]
                    svr_states_clone[action][1] = svr_states_clone[action][1] - [mem]

                    if svr_states_clone[action][0] - [cpu] <= 0 or svr_states_clone[action][1] - [mem] <= 0:
                        # 특정 서버의 자원이 VNF_type 설치에 충분하면 값을 1로
                        theta_0[vnf_type_no][action] = 0
                        pi_0 = simple_convert_into_pi_from_theta(theta_0)
                        pi_3 = np.delete(pi_0[vnf_type_no], pi_svr_to_delete)
                        for i in range(0, q):
                            # pi_2[i]= pi_1[i]/np.sum(pi_1[:])    # 0으로 나누는 경우 경고: RuntimeWarning: invalid value encountered in double_scalars
                            pi_4[i] = pi_3[i] / kahansum(pi_3[:])

                    #print("action, cpu left, mem left", action, svr_states_clone[action][0] - [cpu],
                    #      svr_states_clone[action][1] - [mem])
                    break
                else:
                    ''' policy updation '''
                    theta_0[vnf_type_no][action] = 0
                    pi_0 = simple_convert_into_pi_from_theta(theta_0)
                    pi_3 = np.delete(pi_0[vnf_type_no], pi_svr_to_delete)
                    for i in range(0, q):
                        # pi_4[i]= pi_3[i]/np.sum(pi_3[:])    # 0으로 나누는 경우 경고: RuntimeWarning: invalid value encountered in double_scalars
                        pi_4[i] = pi_3[i] / kahansum(pi_3[:])
                    action = -1

    if action == -1 :  # this condition is introduced to avoid several if conditions, action = -1 means the previous step couldnt allocate the server
        #print("4--Adjacent server cannot be selected -> Attempt to select a datacenter server")
        if (np.sum(pi_6[:])) != 0:
            while (np.sum(pi_6[:])) != 0:   # this loop will make sure that all the datacenter servers are occupied failing the deployment
                if np.random.rand() < epsilon:
                    action = np.random.choice(dc_svr_list, p=pi_6[:])  # 확률 ε로 무작위 행동을 선택함
                else:
                    #svr_to_delete = np.append(local_svr_list, axis=0)
                    #print("(np.where(np.array(pi_4) == 0))[0]",(np.where(np.array(pi_4) == 0))[0])
                    #print("(np.where(np.array(pi_4) == 0))[0]", neigh_svr_list[(np.where(np.array(pi_4) == 0))[0]])
                    svr_to_delete = np.append(local_svr_list,dc_svr_list[(np.where(np.array(pi_6) == 0))[0]], axis=0)
                    #print("svr_to_delete", svr_to_delete)
                    #print("np.delete(Q[sfc_req[-1][aa], :], svr_to_delete)",np.delete(Q[sfc_req[-1][aa], :], svr_to_delete))
                    dc_svr_list1= dc_svr_list[(np.where(np.array(pi_6) != 0))[0]]
                    #print("dc_svr_list1",dc_svr_list1)
                    action = dc_svr_list1[np.nanargmax(np.delete(Q[sfc_req[-1][aa], :], svr_to_delete))]
                    #print("action",action)
                    #action = dc_svr_list[np.nanargmax(np.delete(Q[sfc_req[-1][aa], :], local_svr_list))]
                # print("dc_action=", action)
                if svr_states_clone[action][0] - cpu >= 0 and svr_states_clone[action][
                    1] - mem >= 0:  # action은 선택된 서버 번호+1
                    svr_states_clone[action][0] = svr_states_clone[action][0] - [cpu]
                    svr_states_clone[action][1] = svr_states_clone[action][1] - [mem]

                    if svr_states_clone[action][0] - [cpu] <= 0 or svr_states_clone[action][1] - [mem] <= 0:
                        # 특정 서버의 자원이 VNF_type 설치에 충분하면 값을 1로
                        theta_0[vnf_type_no][action] = 0
                        pi_0 = simple_convert_into_pi_from_theta(theta_0)
                        pi_5 = np.delete(pi_0[vnf_type_no], local_svr_list)
                        for i in range(0, n):
                            # pi_2[i]= pi_1[i]/np.sum(pi_1[:])    # 0으로 나누는 경우 경고: RuntimeWarning: invalid value encountered in double_scalars
                            pi_6[i] = pi_5[i] / kahansum(pi_5[:])

                    #print("action, cpu left, mem left", action, svr_states_clone[action][0] - [cpu],
                    #      svr_states_clone[action][1] - [mem])
                    break
                    # r=0
                else:
                    ''' policy updation '''
                    theta_0[vnf_type_no][action] = 0
                    pi_0 = simple_convert_into_pi_from_theta(theta_0)
                    pi_5 = np.delete(pi_0[vnf_type_no], local_svr_list)
                    for i in range(0, n):
                        # pi_4[i]= pi_3[i]/np.sum(pi_3[:])    # 0으로 나누는 경우 경고: RuntimeWarning: invalid value encountered in double_scalars
                        pi_6[i] = pi_5[i] / kahansum(pi_5[:])
                    action = -1

    if action == -1:
            print("Could not deploy")


    return svr_states_clone, s_a_history,L,r, action, theta_0

def goal_sfc_ret_s_a_Q(Q, epsilon, eta, gamma, line,theta_0,topology):
    svr_states_clone = copy.deepcopy(topology["total_svr_states"])

    actions = np.arange(topology["total_svr_no"])  # 0,1,2,3.....n
    # print("act=", actions)
    # print("---------------------  start of goal_sfc_ret_s_a() --------------------------")

    # print("------ 파일에서 한 라인 씩 처리 시작------------------")
    sfc_req = ast.literal_eval(line)  # Converting string to list
    # print("sfc_req=",sfc_req)


    '''---------------- 참조 정보 계산---------------'''
    L = 0
    r = 0

    t = len(sfc_req)
    # print("sfc_req 길이=",t)
    latency = sfc_req[0]  # 지연 값 추출 저장(추후 사용 예정)

    edge_seq_id = 1  # hardcoding here, as we have taken care of this in function change_edge_sequence(edge):

    total_svr_list = np.arange(topology["total_svr_no"], dtype=int)
    local_svr_list = np.arange(topology["local_svr_no"], dtype=int)
    # print("서버 번호 범위 참조 리스트=", local_svr_list)

    ''' 전체 서버 갯수를 edge_no로 나눈 값에 edge-seq 번호를 이용하여 자국 서버 번호 범위를 추출한다'''
    self_svr_list = np.arange(topology["local_svr_no"] // topology["edge_no"] * (edge_seq_id - 1), topology["local_svr_no"] // topology["edge_no"] * edge_seq_id,
                              dtype='i')
    #print("자국 서버 번호 리스트=", self_svr_list)

    '''전체 서버 번호 행렬에서 자국 서버 번호 범위를 빼면, 결과가 1차원 list로 출력됨'''
    neigh_svr_list = np.delete(local_svr_list, self_svr_list)
    # print("인접국 서버 번호 리스트=", neigh_svr_list)

    dc_svr_list = np.arange(topology["local_svr_no"], topology["total_svr_no"])
    # print(" DC 서버 번호 리스트=", dc_svr_list)
    # ------------- end of 참조 정보 계산------------#

    ### 예: sfc_req x = [10, 1, [4, 1, 3, …]]  vnf_type_no = 0, 1, 3 ....

    sfc_req.pop(0)  # sfc_req[]에서 앞의 2개 원소를 삭제
    sfc_req.pop(0)
    # print("sfc_req_pop=", sfc_req[-1])

    s_a_history = [[sfc_req[0][0], np.nan]]
    # print("hist1=", s_a_history)

    for a in range(topology["total_svr_no"]):
        for b in range(topology["vnf_no"]):
            if svr_states_clone[a][0] >= topology["vnf_type"][b][0] and svr_states_clone[a][1] >= topology["vnf_type"][b][1]:  # 특정 서버의 자원이 VNF_type 설치에 충분하면 값을 1로
                theta_0[b][a] = 1
            else:  # 부족하면 0으로...
                theta_0[b][a] = np.nan
    # 무작위 행동정책 pi_0을 계산하여 이후 학습에 사용함

    for aa in range(0, len(sfc_req[-1])):  # 각 sfc_req의 single list[4, 1, 3...]에 대해서

        svr_states_clone, s_a_history,L,r, action, theta_0= allocate(Q,aa,sfc_req,svr_states_clone,epsilon,local_svr_list,self_svr_list,neigh_svr_list,dc_svr_list, s_a_history,L,r,theta_0,topology)


        '''-------------- s_a_history appending --------------------------------'''

        s_a_history[-1][1] = action
        a = action

        if aa != len(sfc_req[-1]) - 1:  # sfc_req에서 s, s_next를 구하여 s_a_history를 만든다
            s = sfc_req[-1][aa]
            s_next = sfc_req[-1][aa + 1]
            # print("aa=", aa, "state=", s, "next=", s_next)
        else:
            s = sfc_req[-1][aa]
            s_next = 8888

        s_a_history.append([s_next, np.nan])

        # print("s_a_hist=", s_a_history)

        '''======================  SFC 설정 결과의 가공, L (= 노드 지연) 값 추가 ========================'''

        '''---------------------if에 해당하는 경우가 없음. 삭제------------------------'''
        # if len(s_a_history) == 1:     #  s_a_history가  최초 리스트인 [n, nan] 이면
        #     s_a_history[-1][1] = action
        #     s_a_history.append([s_next, np.nan])
        '''-----------------------------------------------------------------------------'''
        if len(s_a_history) >= 3:  # 리스트의 길이가 3 이상일 때에 p_act와 act가 동시 존재한다.
            p_act = s_a_history[-3][1]  # 과거 action을 추출한다, 그리고 자국과 인접국에 대한index를 추출한다.

            '''---------- svr_loc() 함수 호출하여 서버의 위치 정보를 파악하고  L 값을 누적 계산한다 -------------'''
            lc_p_self_ret, lc_p_neigh_ret, lc_p_dc_ret, lc_self_ret, lc_neigh_ret, lc_dc_ret = svr_loc(edge_seq_id,
                                                                                                       p_act, action,topology)
            #print ("server locations :", lc_p_self_ret[0], lc_p_neigh_ret[0], lc_p_dc_ret[0], lc_self_ret[0], lc_neigh_ret[0], lc_dc_ret[0])
            ''' 이전 action이 자국 서버 선택인 경우'''
            # print("self- p_act, action=", p_act, action)   #####

            if lc_p_self_ret[0] != -1 and lc_self_ret[0] != -1:  # (1)번 , 즉 자국 번호 내에  p_act와 act가 있으며,,,,

                ''' 이전 action과 현재 action이 같은 서버이면 노드 지연은 없음 '''
                if p_act == action:
                    # r=10    ##############
                    pass

                ''' 이전 action과 현재 action이 같은 tor에 속하는 서버 중 하나이면'''
                if lc_p_self_ret[0] == lc_self_ret[0] and lc_p_self_ret[1] != lc_self_ret[1]:
                    L = L + 1
                    # r=7
                ''' 이전 action과 현재 action이 자국 내 다른 tor에 속하면'''
                if lc_p_self_ret[0] != lc_self_ret[0]:
                    L = L + 3
                    # r=3
            '''-----------------------------------------------------------'''

            ''' 이전 action이 자국 서버 선택, 현재 action이 인접국 서버의 하나이면'''
            if lc_p_self_ret[0] != -1 and lc_neigh_ret[0] != -1:
                L = L + 7
            '''-------------------------------------------------------------'''

            ''' 이전 action이 인접국 서버 선택, 현재 action이 인접국 서버 선택인 경우'''
            if lc_p_neigh_ret[0] != -1 and lc_neigh_ret[0] != -1:
                ''' 이전 action과 현재 action이 같은 서버이면 노드 지연은 없음'''
                if p_act == action:
                    # r=1
                    pass
                ''' 이전 action과 현재 action이 같은 인접국(edge_no)의 같은 tor에 속하는 다른 서버이면'''
                if lc_p_neigh_ret[0] == lc_neigh_ret[0] and lc_p_neigh_ret[1] == lc_neigh_ret[1] and lc_p_neigh_ret[
                    2] != lc_neigh_ret[2]:
                    L = L + 1

                ''' 이전 action과 현재 action이 같은 인접국(edge_no) 내의 다른 tor에 속하면'''
                if lc_p_neigh_ret[0] == lc_neigh_ret[0] and lc_p_neigh_ret[1] != lc_neigh_ret[1]:
                    L = L + 3

                ''' 이전 action과 현재 action이 서로 다른 인접국에 속하는 경우'''
                if lc_p_neigh_ret[0] != lc_neigh_ret[0]:
                    L = L + 7
            '''-------------------------------------------------------------------'''

            ''' 이전 action이 인접국 서버 선택, 현재 action이 자국 서버 선택이면'''
            if lc_p_neigh_ret[0] != -1 and lc_self_ret[0] != -1:
                L = L + 7
            '''--------------------------------------------------------------------'''

            '''이전 action이 DC 설치 이고 현재 action도  같은 DC 설치인 경우(ppt, 5번)'''

            if len(s_a_history) == 3 and lc_p_dc_ret[0] != -1 and lc_dc_ret[
                0] != -1:  # 최초 action이 DC 설치일때 가입자로부터의 지연(7)을 추가
                if p_act == action:
                    L = L + 7
                ''' 이전 action과 현재 action이 같은 DC의 같은 edge와 같은 tor에 속하는 다른 서버이면'''
                if lc_p_dc_ret[0] == lc_dc_ret[0] and lc_p_dc_ret[1] == lc_dc_ret[1] and lc_p_dc_ret[2] != lc_dc_ret[2]:
                    L = L + 1 + 7
                ''' 이전 action과 현재 action이 같은 DC의 같은 edge의 다른 tor에 속하는 다른 서버이면'''
                if lc_p_dc_ret[0] == lc_dc_ret[0] and lc_p_dc_ret[1] != lc_dc_ret[1]:
                    L = L + 3 + 7
                ''' 이전 action과 현재 action이 같은 DC의 다른 edge에 속하는 서버이면'''
                if lc_p_dc_ret[0] != lc_dc_ret[0]:
                    L = L + 7 + 7

            if len(s_a_history) > 3 and lc_p_dc_ret[0] != -1 and lc_dc_ret[0] != -1:
                if p_act == action:
                    pass
                ''' 이전 action과 현재 action이 같은 DC의 같은 edge와 같은 tor에 속하는 다른 서버이면'''
                if lc_p_dc_ret[0] == lc_dc_ret[0] and lc_p_dc_ret[1] == lc_dc_ret[1] and lc_p_dc_ret[2] != lc_dc_ret[2]:
                    L = L + 1
                ''' 이전 action과 현재 action이 같은 DC의 같은 edge의 다른 tor에 속하는 다른 서버이면'''
                if lc_p_dc_ret[0] == lc_dc_ret[0] and lc_p_dc_ret[1] != lc_dc_ret[1]:
                    L = L + 3
                ''' 이전 action과 현재 action이 같은 DC의 다른 edge에 속하는 서버이면'''
                if lc_p_dc_ret[0] != lc_dc_ret[0]:
                    L = L + 7
            '''이전 action이 자국 또는 인접국  설치 이고 현재 action이 DC 설치인 경우'''
            if (lc_p_self_ret[0] != -1 and lc_dc_ret[0] != -1) or (lc_p_neigh_ret[0] != -1 and lc_dc_ret[0] != -1):
                L = L + 9
            '''이전 action이 DC  설치 이고 현재 action이 자국 또는 인접국 설치인 경우'''
            if (lc_p_dc_ret[0] != -1 and lc_self_ret[0] != -1) or (lc_p_dc_ret[0] != -1 and lc_neigh_ret[0] != -1):
                L = L + 9

    ''' ------------------------- SFC의 자원 요구량 계산 --------------------------------'''
    cpu_req = 0
    mem_req = 0
    for i in range(0, len(sfc_req[-1])):
        cpu_req = cpu_req + topology["vnf_type"][sfc_req[-1][i]][0]
        mem_req = mem_req + topology["vnf_type"][sfc_req[-1][i]][1]
    #print("cpu_req mem_req=", cpu_req, mem_req)

    cpu_ratio = cpu_req / topology["cpu_no"]
    mem_ratio = mem_req / topology["mem_no"]  ###### 수정_0428

    '''======----------------------- Reward 부여하기------=========================='''
    '''----- 다시 s_a_history에서 Q 함수에 넘겨줄 값(s, a, s_next)을 추출하고, 누적된 L 값으로 구한 reward와 함께 Q함수를 실행한다-----'''
    '''----- 즉, 하나의 SFC를 구성하는 모든 s, a, a_next에 동일한 reward를 적용한다'''
    ''' s_a_history 예==[[1, 15], [2, 14], [3, 15], [2, 10], [3, 15], [8888, nan]] '''

    hist_len = len(s_a_history) - 1
    for i in range(hist_len):  # 맨끝에 [8888, nan]이 추가되었므로 -1을 취함
        s = s_a_history[i][0]  # s, a, s_next를 추출하여 하나의 SFC를 구성하는 (s, a,)에 동일한 reward를 부여하한다.
        a = s_a_history[i][1]
        s_next = s_a_history[i + 1][0]

        #r= 50 * (cpu_ratio+mem_ratio/2) * math.exp(-2*L/hist_len)
        r = 50 * (cpu_ratio + mem_ratio / 2) * math.exp(-2 * L/hist_len)

        Q = Q_learning(s, a, r, s_next, Q, eta, gamma)  # fow loop 안에서 (s,a, s_next)에 대해실행


    #print("(state, action)의 history 와 Latency=", s_a_history, L,r)

    return s_a_history, Q, L, latency, r

def sfc_edge(simul):
    SIMULATION = simul
    file = open("./sfc_config.txt", 'r')  # 파일에서 하나의 sfc-req를 읽어 들인다.
    output_length = open("./log-Delay.csv", 'w')
    output_reward  = open("./log-Reward.csv", 'w')
    topology = {
        "core_switch_id_for_edge":"",
        "edge_no":0,
        "tor_no":0,
        "svr_no":0,
        "cpu_no":0,
        "mem_no":0,
        "core_switch_id_for_DC":"",
        "dc_edge_no":0,
        "dc_tor_no":0,
        "dc_svr_no":0,
        "svr":[],
        "svr_states":[],
        "dc_svr":[],
        "dc_svr_states":[],
        "total_svr_states":[],
        "vnf_flav":[],
        "vnf_type":[],
        "vnf_names":[],
        "vnf_no":0,
        "vnf_prefix":"",
        "local_svr_no":0,
        "self_svr_no":0,
        "neigh_svr_no":0,
        "dc_local_svr_no":0,
        "total_svr_no":0,
        "load_these_servers":[],
        "edge_seq":0
    }


    while True:
        line = file.readline()  # put multiple tests in the test file. each line is considered as one SFC request
        #print("line", line)
        if not line:
            print("check sfc_config.txt file")
            break
        for N in range(1): # this loop can be configured to run one test multiple times. replace 1 with 10 to run this SFC request 10 times
            # print("svr_states=",svr_states)
            # This line can be used to hard code each local server capacity.
            if (SIMULATION):
                topology["edge_no"] = 3  # Edge switch의 수 #2
                topology["tor_no"] = 3  # edge swtch 별 tor swtch의 수 #1
                topology["svr_no"] = 3  # TOR에 연결된 서버의 갯수 #2 can have seperate number of svr_no for each tor
                topology["cpu_no"] = 6  # 서버 내 vcpu 수        #assign seperately by iteration
                topology["mem_no"] = 8  # 서버 내 메모리 량       #assign seperatel by iteration

                # edge 서버 자원 정의
                topology["dc_edge_no"] = 3  # Edge switch의 수
                topology["dc_tor_no"] = 3  # edge swtch 별 tor swtch의 수
                topology["dc_svr_no"] = 3  # TOR에 연결된 서버의 갯수
                topology["dc_cpu_no"] = 32  # 서버 내 vcpu 수
                topology["dc_mem_no"] = 48  # 서버 내 메모리 량


                # This line can be used to hard code each local server capacity.
                #svr_states = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                #              [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                #              [0, 0], [4, 8], [4, 8]]
                # print("svr_states1=", svr_states)
                topology["vnf_type"] = np.array([[1, 1],  # vnf0은 cpu 1 개, mem 1 GB 사용
                                     [1, 2],  # vnf1은 cpu 1 개, mem 2 GB 사용....
                                     [2, 2],
                                     [2, 4],
                                     [3, 3],
                                     [3, 6],
                                     [4, 4],
                                     [4, 8]])  # VNF 7은 cpu 4개, mem 8GB 사용

                topology["local_svr_no"] = topology["edge_no"] * topology["tor_no"] * topology["svr_no"]  #
                topology["self_svr_no"] = topology["local_svr_no"] / topology[
                    "edge_no"]  # dtype == float 이 되므로 주의, 아래도...
                topology["neigh_svr_no"] = topology["local_svr_no"] - topology["self_svr_no"]

                topology["dc_local_svr_no"] = topology["dc_edge_no"] * topology["dc_tor_no"] * topology["dc_svr_no"]
                # dc_self_svr_no = dc_local_svr_no/dc_edge_no          # dtype == float 이 되므로 주의, 아래도...
                # dc_neigh_svr_no = dc_local_svr_no - dc_self_svr_no

                topology["total_svr_no"] = topology["local_svr_no"] + topology["dc_local_svr_no"]
                # set all the topology variables as you like to simulate

                topology["svr_states"] = np.zeros((topology["local_svr_no"], 2)) + [topology["cpu_no"], topology[
                    "mem_no"]]  # ---> 호출 문제로 함수 내부로 이전
                topology["dc_svr_states"] = np.zeros((topology["dc_local_svr_no"], 2)) + [topology["dc_cpu_no"],
                                                                                          topology[
                                                                                              "dc_mem_no"]]  # ---> 호출 문제로 함수 내부로 이전
            else:
                # get the network topology
                # set all the topology variables
                topology = get_topology_svr_vnf(topology)
                seperator = ","
                final_str = seperator.join([str(item) for item in topology['vnfs']])
                #print(final_str)
                line = "[10" + ","+str(topology["edge_seq"])+"," + "[" + final_str + "]]"
                #svr=topology["svr"]
                #svr_states = topology["svr_states"]

                #dc_svr = topology["dc_svr"]
                #dc_svr_states = topology["dc_svr_states"]

                #vnf_flav = topology["vnf_flav"]

                #vnf_type = topology["vnf_type"]
                #vnf_name= topology["vnf_name"]


                topology["local_svr_no"] = int(topology["edge_no"]) * int(topology["tor_no"]) * int(topology["svr_no"])  #
                topology["self_svr_no"] = topology["local_svr_no"] / int(topology["edge_no"])  # dtype == float 이 되므로 주의, 아래도...
                topology["neigh_svr_no"] = topology["local_svr_no"] - topology["self_svr_no"]

                topology["dc_local_svr_no"] = int(topology["dc_edge_no"]) * int(topology["dc_tor_no"]) * int(topology["dc_svr_no"])
                # dc_self_svr_no = dc_local_svr_no/dc_edge_no          # dtype == float 이 되므로 주의, 아래도...
                # dc_neigh_svr_no = dc_local_svr_no - dc_self_svr_no

                topology["total_svr_no"] = topology["local_svr_no"] + topology["dc_local_svr_no"]
                # set all the topology variables as you like to simulate


            #print("svr_states=", topology["svr_states"])
            #print("dc_svr_states=",topology["dc_svr_states"])
            # print("svr_states=",svr_states)
            # print("dc_svr_states=",dc_svr_states)

            topology["total_svr_states"] = np.append(topology["svr_states"], topology["dc_svr_states"], axis=0)  # np.append()로 2개의 array를 통합함
            # print(total_svr_states)

            topology["vnf_no"] = len(topology["vnf_type"])

            # Q러닝 알고리즘으로 SFC 설치하기

            # 계산 시간 출력 용

            '''-------------------------------------------------------------------------------------------------------'''
            ## 정책 theta_0 의 초기값 생성
            ''' theta_0 를 0으로 초기화하여 생성하고,....  
            states 행렬의 서버 별 자원 상태와 vnf_type 행렬의 자원 요구량을 비교하여
            자원이 충분하면 theta_0의 각원소를 1로 설정하고 부족하면 설치 불가하므로 0으로 설정한다.'''

            '''상태는 VNF의 자원요구량에 의해 상태변화를 야기하는 것으로... 
            액션은 자원이 가용한 어떤 서버를 선택하여 설치하는 것으로 정의했다'''

            theta_0 = np.zeros((int(topology["vnf_no"]), int(topology["total_svr_no"])))  # 행= VNF 종류 갯수,  열 = 전체 서버 수
            # print(theta_0)


            # 행동가치 함수 Q의 초기 상태

            [a, b] = theta_0.shape  # 열과 행의 갯수를 변수 a, b에 저장
            Q = np.random.rand(a, b) * theta_0  # theta_0를 곱한 이유는 nan을 반영하기 위한 것
            #print(Q)

            start = time.perf_counter()

            eta = 0.01  # 학습률
            gamma = 0.5  # 시간할인율 = 0.9
            epsilon = 0.8  # ε-greedy 알고리즘 epsilon 초깃값=0.5
            v = np.nanmax(Q, axis=1)  # 각 상태마다 가치의 최댓값을 계산
            is_continue = True
            episode = 1

            L_list = []  # return되는 L 값을 저장할 리스트 생성
            r_list = []

            # Q- Learning starts from here
            while is_continue:  # is_continue의 값이 False가 될 때까지 반복
                print("**************** 에피소드: " + str(episode), "******************")

                # ε 값을 조금씩 감소시킴
                epsilon = epsilon /1.05  # 2

                # Q러닝 실행 후, 결과로 나온 행동 히스토리와 Q값을 변수에 저장
                s_a_history, Q, L, latency,r = goal_sfc_ret_s_a_Q(Q, epsilon, eta, gamma, line, theta_0,topology)

                '''---------- L 값을 리스트에 저장한다'''
                L_list.append(L)
                r_list.append(r)
                #print("L_list=", L_list)
                #print("L_list=", r_list)
                if epsilon > 0.1:  # 무작위 선택을 하는 동안 min, max를 구한다.
                    #print("epsilon",epsilon)
                    max_num = max(L_list)
                    min_num = min(L_list)
                    avg = np.mean(L_list)
                    # print("Min Max Avg = ",min_num, max_num, avg)

                '''-----------------------------------'''
                #    reward(1,1)

                #print("SFC에 설치한 VNF 수는 " + str(len(s_a_history) - 1) + " 개이고, 노드 지연은", L, "입니다")

                # 에피소드 반복

                print("s_a_history:", s_a_history)
                print("L_list", L_list)


                episode = episode + 1
                if episode > 100:  # 100
                    break

            end = time.perf_counter()  # 계산 시간 출력

            resource=0
            for j in range(len(s_a_history)-1):
                resource=resource+ topology["vnf_type"][s_a_history[j][0]][0] + topology["vnf_type"][s_a_history[j][0]][1]
            resource=resource/2
            end = time.perf_counter()  # 계산 시간 출력
            output_length.write(str(L_list))
            output_length.write("\n")
            output_reward.write(str(len(s_a_history)-1 )+ "," + str(resource) + "," + str(r_list) + "," + str(end - start))
            output_reward.write("\n")

    #print(s_a_history)
    #print(L_list)

    print("")
    file.close()
    output_length.close()
    output_reward.close()
    '''------ latency를 이용한 종료 조건 설정--------'''
    '''action의 선택은 epsilon 값에 따라 무작위로 
    
    결정되기도 하므로 무작위 결정에 대응하여 latency를 도입함.'''

    start1 = time.perf_counter()
    if (SIMULATION == False):  # This means this is not simulation but a real deployment
        #print("vnf_flav", topology["vnf_flav"])
        #print("svr ", topology["svr"])
        #print("svr ", )
        #deploy_sfc(s_a_history,topology["vnf_flav"], topology["svr"],topology["dc_svr"],local_svr_no,dc_local_svr_no,vnf_name,topology["vnf_prefix"])
        deploy_sfc(s_a_history, topology)
    end1 = time.perf_counter()
    print("-----------------Time taken-------------------------")
    print("Q-러닝 실행 시간 =", end - start, "seconds")
    print("SFC Deployment =", end1 - start1, "seconds")


def main(argv):
    # Call this function with "True" for simulation only and "False" for actually test bed usage and deployment
    sfc_edge(False)

if __name__ == "__main__":
    #print(len(sys.argv), sys.argv)
    main(sys.argv)