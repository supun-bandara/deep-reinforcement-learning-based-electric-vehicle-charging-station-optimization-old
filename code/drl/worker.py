import torch.multiprocessing as mp
from util.utils import v_wrap, set_init, push_and_pull, record
from environment.env import ENV
from drl.net import Net
import numpy as np
import torch

UPDATE_GLOBAL_ITER = 5 # Number of global iterations before updating the global network
GAMMA = 0.9 # Discount factor for future rewards in the reinforcement learning algorithm
max_charging_rate=5 #Unit: 20 KWh
price_upper_bound=6
N_A=max_charging_rate+price_upper_bound # dimension of action space
N_S=6 # dimension of state space (number of features in state)
MAX_EP = 2 #3000 # Maximum number of episodes for training
MAX_EP_STEP = 10 #200 # Maximum number of steps per episode

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        environment = ENV()
        self.env = environment.env

    def run(self):
        #print("worker - run - Starting " + self.name)
        total_step = 1
        while self.g_ep.value < MAX_EP:
            #print("")
            #print(f"worker - run - Starting - self.g_ep: {self.g_ep}")
            ################################
            #       Initial State          #
            ################################
            a=np.array([100,0])
            s=torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0]).reshape((1,N_S)).unsqueeze(0)
            real_state=np.array([])
            #########################
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                #print(f"worker - run - Step : {t}")
                a = self.lnet.choose_action(s)
                r, real_state_, s_= self.env(a,real_state,t) # calling env
                r=np.expand_dims(np.expand_dims(r, 0), 0)
                s_=s_.reshape((1,N_S)).unsqueeze(0).float()
                ep_r += r
                buffer_a.append(np.array(a))
                buffer_s.append(s.squeeze().numpy())
                buffer_r.append(r.squeeze())
                done=False
                if t == MAX_EP_STEP - 1:
                    done = True
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    
                    if done:  # done and #print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                real_state=real_state_
                total_step += 1
        self.res_queue.put(None)