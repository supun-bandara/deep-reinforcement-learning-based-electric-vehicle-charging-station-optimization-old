import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.utils import v_wrap, set_init, push_and_pull, record

# Logits are the outputs of a neural network before the activation function is applied

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        #print(f"Net - s_dim: {s_dim}, a_dim: {a_dim}")
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128) # policy network
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128) # value network
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical
        self.max_charging_rate=5 #Unit: 20 KWh

        #torch.nn.init.xavier_uniform_(self.pi1)
        #torch.nn.init.xavier_uniform_(self.pi2)
        #torch.nn.init.xavier_uniform_(self.v1)
        #torch.nn.init.xavier_uniform_(self.v2)

    def forward(self, x):
        #print(f"Net - forward - x: {x}")
        ##print(x.shape)
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        #print(f"Net - forward - return logits: {logits}, values: {values}")
        return logits, values

    def choose_action(self, s): # choose action for the state
        #print(f"Net - choose_action - s: {s}")
        self.eval()
        logits, _ = self.forward(s)
        #print(f"logits[0][0][:self.max_charging_rate] : {logits[0][0][:self.max_charging_rate]}")
        #print(f"logits[0][0][self.max_charging_rate:] : {logits[0][0][self.max_charging_rate:]}")
        prob1 = F.softmax(logits[0][0][:self.max_charging_rate], dim=-1).data
        prob2 = F.softmax(logits[0][0][self.max_charging_rate:], dim=-1).data
        #print(f"Net - choose_action - prob1: {prob1}, prob2: {prob2}")
        m1 = self.distribution(prob1)
        m2=self.distribution(prob2)
        #print(f"Net - choose_action - m1: {m1}, m2: {m2}")
        a1=m1.sample().numpy()
        a2=m2.sample().numpy()
        #print(f"Net - choose_action - return a1: {a1}, a2: {a2}")
        return np.array([a1,a2]) # return action

    def loss_func(self, s, a, v_t):
        #print(f"Net - loss_func - s: {s}, a: {a}, v_t: {v_t}")
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        ##print(f"logits[:,:self.max_charging_rate] : {logits[:,:self.max_charging_rate]}")
        ##print(f"logits[:,self.max_charging_rate:] : {logits[:,self.max_charging_rate:]}")
        prob1 = F.softmax(logits[:,:self.max_charging_rate], dim=-1).data
        prob2 = F.softmax(logits[:,self.max_charging_rate:], dim=-1).data    
        m1 = self.distribution(prob1)
        m2=self.distribution(prob2)
        ##print(f"a.shape : {a.shape}")
        ##print(f"a[:,0] : {a[:,0]}")
        ##print(f"a[:,1] : {a[:,1]}")
        exp_v = m1.log_prob(a[:,0])*m2.log_prob(a[:,1])* td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean() # c_loss = value loss and a_loss = policy loss
        #print(f"Net - return total_loss : {total_loss}, c_loss: {c_loss}, a_loss: {a_loss}")
        return total_loss