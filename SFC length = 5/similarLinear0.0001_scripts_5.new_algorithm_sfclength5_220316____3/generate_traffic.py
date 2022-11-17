
import subprocess
import json
import pandas as pd


# import datetime as dt
# end_time = dt.datetime.now()
# this script is to run the traceroute command for SFC, the SFC configuration are taken from input.json

with open('input.json') as f:
    sfc = json.load(f)
    username=sfc["username"]    #username of client machine for generating traffic
    password=sfc["password"]    #password of Client machine for generating trafffic
    ip_src=sfc["src_ip_prefix"][:-3] #Ip adress of cline
    ip_dst=sfc["dst_ip_prefix"][:-3] #Ip adrress of desitnation
    client=username+"@"+ip_src
    traffic_pattern=sfc["traffic_pattern_file"] # Abline traffic pattern file
    connections=sfc["-c_arg_wrk"]  # number of sessions to generate traffic
    duration=sfc["-d_arg_wrk"]     # number of seconds to generate each pattern
    traffic_scaler = sfc["-R_wrk_multiplier"] # Abline traffic is normalized from 1~10, this faction will multiply each read
    output_latency=sfc["latency_result_file"] # After generating traffic record the delay measure of each request

    df = pd.read_csv(traffic_pattern,
                     header=None, sep=',')
    df.columns = ['TIME', 'TRAFFIC']
    target_names = ['TRAFFIC']
    shift_days = 1
    shift_steps = shift_days * 3  # Number of hours.
    df_targets = df[target_names].shift(-shift_steps)

    x_data = df.values[0:-shift_steps]
    y_data = df_targets.values[:-shift_steps]

    #print(username, password, ip_src,ip_dst)
    print("==================================================================")
    print("=    Traffic Generation tool : Client- Wrk2, Server -nginx       =")
    print("==================================================================")
    print()

    http_connection="http://"+ip_dst+":80/index.html"
    print(http_connection)
    output_latency = open(output_latency,'w')
    responsetime = 0
    for i in range(len(y_data)):
        dur=int(duration)  # -responsetime
        command="wrk -d" + str(dur)+"s" + " -c" + str(connections) + " -t2" + " -R" + str(int(y_data[i][0]*int(traffic_scaler)))

        for i in range(3):  # generate each traffic for 3 minutes
            result = subprocess.run(["sshpass", "-p", password, "ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", client, "sudo", command , http_connection], stdout = subprocess.PIPE, stderr = subprocess.STDOUT,)

        #print(str(result))

            list = str(result).split()

            print(list)
            for i in range(len(list)):
                if (list[i] == "Latency"):
                    print(list[i + 1], list[i+2], list[i+3])
                    output_latency.write(list[i + 1]+","+list[i+2]+","+ list[i+3]+",")
                    # We must reduce this response time from the total traffic duration time.
                    # this is becaouse for this much of the time there was no traffic generated
                    # and to syncronize traffic generation and keep overall traffic generation time in the limit we must take this action
                    if list[i+1].find("ms") and list[i+1].find("us") == -1:
                        responsetime = int(list[i+1][:list[i+1].find(".")])

                if (list[i] == "responses:"):
                    print(list[i+2] + "," + list[i+3])
                    output_latency.write(list[i + 2] + "," + list[i + 3])

            output_latency.write("\n")

    output_latency.close()
