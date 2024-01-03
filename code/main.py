# http://www.energyonline.com/Data/GenericData.aspx?DataId=20

from drl.net import Net
from drl.worker import Worker
import torch.multiprocessing as mp
from drl.shared_adam import SharedAdam  
import gym
import os
import random
os.environ["OMP_NUM_THREADS"] = "1"

max_charging_rate=5 #Unit: 20 KWh
price_upper_bound=6
N_A=max_charging_rate+price_upper_bound # dimension of action space
N_S=6 # dimension of state space (number of features in state)

if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network    
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # unparallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(1)]
    # parallel training
    #workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
    # #print with different colours
    ##print(res)
    #plt.plot(res)
    #plt.show()
