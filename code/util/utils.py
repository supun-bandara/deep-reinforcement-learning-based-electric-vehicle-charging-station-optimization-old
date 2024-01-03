"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np

# this function converts a NumPy array to a PyTorch tensor and ensures that the data type matches the specified dtype (default is np.float32).
def v_wrap(np_array, dtype=np.float32): 
    #print(f"utils - v_wrap - np_array: {np_array}, dtype: {dtype}")
    if np_array.dtype != dtype:
            try:
                    np_array = np_array.astype(dtype)
            except:
                return np_array.float()
    return torch.from_numpy(np_array)

# initializes the weights of the layers in a neural network using normal distribution for weights and setting biases to zero.
def set_init(layers): 
    #print(f"utils - set_init - layers: {layers}")
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


# def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
#     if done:
#         v_s_ = 0.               # terminal
#     else:
#         v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

#     buffer_v_target = []
#     for r in br[::-1]:    # reverse buffer r
#         v_s_ = r + gamma * v_s_
#         buffer_v_target.append(v_s_)
#     buffer_v_target.reverse()

#     loss = lnet.loss_func(
#         v_wrap(np.vstack(bs)),
#         v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
#         v_wrap(np.array(buffer_v_target)[:, None]))

#     # calculate local gradients and push local parameters to global
#     opt.zero_grad()
#     loss.backward()
#     for lp, gp in zip(lnet.parameters(), gnet.parameters()):
#         gp._grad = lp.grad
#     opt.step()

#     # pull global parameters
#     lnet.load_state_dict(gnet.state_dict())

# implements the push-and-pull mechanism used in the A3C algorithm to update the global model parameters with the gradients computed from the local model.
def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    #print(f"utils - push_and_pull - opt: {opt}, lnet: {lnet}, gnet: {gnet}, done: {done}, s_: {s_}, bs: {bs}, ba: {ba}, br: {br}, gamma: {gamma}")
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    #print(np.vstack(bs))
    #print(np.vstack(ba))
    #print(buffer_v_target)
    #print(np.vstack(bs))
    to_loss_s=v_wrap(np.vstack(bs))
    to_loss_a= v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba))
    #to_loss_b=v_wrap(np.array(buffer_v_target)[:, None])

    to_loss_b=v_wrap(np.vstack(br))

    #print(to_loss_s.shape)
    #print(to_loss_a.shape)
    #print(to_loss_b.shape)


    loss = lnet.loss_func(to_loss_s,to_loss_a,to_loss_b)

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

# updates and prints information about the training progress, such as the episode number and the moving average of episode rewards.
def record(global_ep, global_ep_r, ep_r, res_queue, name):
    #print(f"utils - record - global_ep: {global_ep}, global_ep_r: {global_ep_r}, ep_r: {ep_r}, res_queue: {res_queue}, name: {name}")
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )