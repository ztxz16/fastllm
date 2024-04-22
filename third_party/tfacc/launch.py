import os
import threading
import sys

def run(cmd):
    os.system(cmd)

#modelName = "/root/flmModels/chatglm3-6b-int4.flm"
#modelName = "/root/flmModels/qwen1.5-36B-chat-int4.flm"

total = int(sys.argv[1])

for i in range(total):
    st = i * 40
    end = st + 39
    cmd = "numactl -C " + str(st) + "-" + str(end) + " -m " + str(i) + " ./server " + str(i) + " " + str(total)
    print(cmd)
    (threading.Thread(target = run, args = ([cmd]) )).start()




