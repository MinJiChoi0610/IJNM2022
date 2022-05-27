# <traceroute_test-with-path-delay.py>

import subprocess
import json

import ni_mon_client
from config import cfg
from ni_mon_client.rest import ApiException
import ni_nfvo_client
from ni_nfvo_client.rest import ApiException
with open('input.json') as f:
    sfc = json.load(f)
    username=sfc["username"]
    password=sfc["password"]
    ip_src=sfc["src_ip_prefix"][:-3]
    ip_dst=sfc["dst_ip_prefix"][:-3]
    client=username+"@"+ip_src

    #print(username, password, ip_src,ip_dst)
    print("=======================================================")
    print("=             Latency calculation tool : TRACEROUTE               =")
    print("=======================================================")
    print()

    #file = open("./traceroute_result.txt", 'w')
    result= subprocess.run(["sshpass", "-p", password, "ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         client, "sudo", "traceroute", ip_dst], stdout = subprocess.PIPE, stderr = subprocess.STDOUT,)
    #print(result.stdout.decode("utf-8"),type(result.stdout.decode("utf-8")),"Suman Pandey")
    #file = open("plot_graph.txt","w")
    #file.write(result)
    #node=[]
    vnf_flav =str(result.stdout.decode("utf-8")).split(" ")
    #print(len(vnf_flav))
    trace={
        "ip" : [],
        "avg_delay":[]
    }
    for i in range(len(vnf_flav)):
        #print(vnf_flav[i] + "****")
        if vnf_flav[i] in ["1","2","3","4","5","6","7","8"] :
            #print("Encounterd :", vnf_flav[i])
            seq=int(vnf_flav[i])-1
            if (vnf_flav[i+2]!="*"):
                trace["ip"].append(vnf_flav[i+2])
                n=5
            else:
                trace["ip"].append(vnf_flav[i + 3])
                n=6

            delay=0
            #print("vnf_flav[i+5]",vnf_flav[i+n])

            if vnf_flav[i+n] !="*":
                delay= float(vnf_flav[i+n])
            else:
                if vnf_flav[i + n+1] != "*":
                    delay = float(vnf_flav[i +n+1])
                else:
                    if vnf_flav[i + n + 2] != "*":
                        delay = float(vnf_flav[i + n+2])
            trace["avg_delay"].append(delay)

    print("***********latency from Client to each VNF*************")
    #print(trace)
    l = len(trace["ip"])
    #print("length", l)

    for i in range(l-1):
        print(ip_src +"(client)" + "----" +str(trace["avg_delay"][i]/2) + " ms " + "----" + trace["ip"][i] + "(vnf" + str(i+1)+")")
    print(ip_src + "(client)" + "----" + str(trace["avg_delay"][l-1]/2) + " ms " + "----" + trace["ip"][l-1] + "(destination)")

    print()
    print("**************Calculated latency of SFC path**************")

    latency=0
    calculatedlatency=0
    print(ip_src + "(client)" + "--->", end="")
    for i in range(l-1):
        calculatedlatency= format(abs(trace["avg_delay"][i]/2 - latency),'.2f')
        print(str(calculatedlatency)+ " ms " + "--->" + trace["ip"][i] + "(vnf" + str(i+1) +")" + "--->",end="")
        latency= trace["avg_delay"][i]/2
    calculatedlatency = format(abs(trace["avg_delay"][l-1]/2 - latency), '.2f')
    print(str(calculatedlatency) + " ms " + "---->" + trace["ip"][l - 1] + "(destination)")

    print()
    print("*****************Overall Latency via SFC******************")

    print(str(trace["avg_delay"][l-1]/2) + " ms ")
    with open('trace.json', 'w') as fp:
        json.dump(trace, fp)
