import numpy as np
import noisy_reservoir
import pickle
import os
import time

# In this task, the reservoir is trained to behave as a sort of sequential digital-to-analog converter
# the inputs are sequential 3-bit codes,
# in which zero bits are coded as a negative half sine, stretched across input_len samples, and
# one bits are coded as a positive half sine, stretched across input_len samples,
# after the entire code has been observed, the network should output one of 8 analog levels in [-1,1],
# and keep this level constant for input_len/2 time steps

# set number of training iterations (originally = 1000000)
num_training_iterations=1000000
dump_cycle_time=100000
io_cycle_time=5000

filebase='res_3bits_andrea_01_'

do_continue=0
iterations = 0
input_len = 10


#TODO keep in mind average reward over all tasks!!
def compute_reward(input_val, obs, target_neurons=[42, 59]):  # just two randomly selected target neurons
    # select desired output for chosen input pattern
    v = outputs[input_val]

    diff = (v - obs[-input_len / 2:, target_neurons].sum(1))  # messy:  global variable input_len???
    err = np.mean(diff ** 2)  #MSE
    return -err


def do_iteration(reservoir, input_val, U, mean_reward, learn, alpha, noise=0.01, norm=None, mask=None, rmask=None):
    X, Z, Y, S = reservoir.run(U, noise, mask=rmask)
    reward = compute_reward(input_val, S)
    if (learn):
        deltaW = reservoir.learn(reward, mean_reward, Z, X, alpha, norm, mask)
    else:
        deltaW = np.zeros(reservoir.W.shape)
    return reward, deltaW, S

if do_continue:
    files=os.listdir(".")
    numfiles=len(files)
    for i in range(numfiles):
        if files[i][:17]=="res_3bits_andrea_":
            iterations=max(iterations,int(files[i][17:-4]))
    if iterations>0:
        print "continuing from previous session at iteration "+str(iterations+1)+"\n"
        r = pickle.load(open("res_3bits_andrea_"+str(iterations)+".pkl"))
        res=r["res"].copy()
        inputs=r["inputs"].copy()
        outputs=r["outputs"].copy()
        W0=r["W0"].copy()
        best_W=r["best_W"].copy()
        input_hist=np.ndarray.tolist(r["input_hist"])
        max_reward=r["max_reward"]
        r_hist=np.ndarray.tolist(r["r_hist"])
        r_hist_mean=np.ndarray.tolist(r["r_hist_mean"])
        sel_probs=np.ndarray.tolist(r["sel_probs"])
if iterations == 0:
    #create input sequence
    input_1 = np.sin(np.linspace(0, np.pi, input_len))
    input_0 = -input_1

    input_001 = np.hstack((input_0, input_0, input_1))
    input_101 = np.hstack((input_1, input_0, input_1))
    input_100 = np.hstack((input_1, input_0, input_0))
    input_000 = np.hstack((input_0, input_0, input_0))
    input_011 = np.hstack((input_0, input_1, input_1))
    input_111 = np.hstack((input_1, input_1, input_1))
    input_010 = np.hstack((input_0, input_1, input_0))
    input_110 = np.hstack((input_1, input_1, input_0))

    inputs = np.vstack((input_001, input_100, input_101, input_000, input_011, input_111, input_010, input_110))
    outputs = np.linspace(-1, 1, 8)  #np.array([1,1,0,0,0,1,1,0])#np.array([-1,1,1,-1])

    # create a random recurrent neural network
    res = noisy_reservoir.FastNoisyReservoir(1, 100, spec_rad=1.15)

    r = []
    r_hist = []
    r_hist_mean = []
    input_hist = []

    o = np.random.randn(res.W[:, :2].shape[0], res.W[:, :2].shape[1])
    o[:, 0] *= 1.0
    o[:, 1] *= 1.0


    res.W[:, :2] = np.where(np.random.random(o.shape) > 0.8, o, 0.)
    res.W[42, :2] = 0.
    res.W[59, :2] = 0.

    max_reward = -100
    W0 = res.W.copy()

    sel_probs = []

Wmask = np.ones(res.W.shape)
Wmask[[42, 59]] = 0
rmask = np.ones(res.W.shape[0])
rmask[[42, 59]] = 0
rmask[::2] = 0

prob_r = np.array([0, 1, 2, 3, 4, 5, 6, 7])  #np.array([0,1,1,1,1,2,2,2,2,2,3,3,3,3,1])#

mean_len = 200

for i in range(iterations,num_training_iterations,1):
    #res.run(input_111.reshape((-1,1)),0,mask=rmask)
    tic=time.time()
    iterations += 1
    prob_r_c = prob_r.copy()
    np.random.shuffle(prob_r_c)
    input_val = prob_r_c[0]
    #choose a new target
    #compute avg difference between mean reward and reward per input and then
    sel_prob = np.zeros(8)
    task_mean_reward = np.zeros(8)
    for j in xrange(8):
        rm = np.array(r_hist_mean[-mean_len:])[np.where(np.array(input_hist[-mean_len:]) == j)[0]]

        task_mean_reward[j] = rm.mean()
        r = np.array(r_hist[-mean_len:])[np.where(np.array(input_hist[-mean_len:]) == j)[0]]
        if (rm.shape[0] <= 3):
            sel_prob[j] = 1.
        else:
            sel_prob[j] = np.mean(np.abs(r - rm))

    #if(i%1000==0):
    #	for j in xrange(8):
    #		if(-task_mean_reward[j]>0.05):
    #			break
    #	if(j<3):
    #		prob_r = np.arange(3)
    #	else:
    #		prob_r = np.arange(j+1)

    #sel_prob=-task_mean_reward
    #sel_prob[np.where(np.isnan(sel_prob))[0]]=1
    sel_probs.append(sel_prob)
    sel_res = sel_prob.copy()
    for j in xrange(1, 8):
        sel_res[j] = sel_res[j - 1] + sel_res[j]
    sel_rand = np.random.random() * sel_res[-1]
    #for j in xrange(4):
    #	if(sel_rand<sel_res[j]):
    #		input_val = j
    #		break

    #compute mean_reward
    reward_hist = np.array(r_hist[-mean_len:])[np.where(np.array(input_hist[-mean_len:]) == input_val)[0]]
    if (reward_hist.shape[0] > 3):
        mean_reward = np.mean(reward_hist)
        learn = True
    else:
        mean_reward = np.mean(reward_hist)
        if (np.isnan(mean_reward)):
            mean_reward = 0.
        learn = False

    input_seq = inputs[input_val].reshape((-1, 1))

    reward, deltaW, S = do_iteration(res, input_val, input_seq, mean_reward, learn, .01, 0.05, None, Wmask,
                                     rmask)  #learning rate 0.01, noise 0.05
    r_hist.append(reward)
    r_hist_mean.append(mean_reward)
    input_hist.append(input_val)
    if (max_reward < reward):
        best_W = res.W.copy()
        max_reward = reward
    #print np.sqrt(np.sum(deltaW**2,1))
    if (iterations % io_cycle_time == 0):
        toc=time.time()
        print "\n \r iteration: %d \treward %f \tmean_reward %f \t mean abs W %f \t max abs deltaW %f input: %d" % (
            iterations, reward, mean_reward, np.abs(res.W[:, 3:]).mean(), np.max(np.abs(deltaW)), input_val)
        print "sel prob: %s" % str(sel_prob)
        print "mean task reward: %s" % str(np.round(task_mean_reward, 5))
        print "mean task reward sqrt: %s" % str(np.round(np.sqrt(-task_mean_reward), 5))
        print "prob_r: %s" % str(prob_r)
        print "spec rad: %f" % np.max(np.abs(np.linalg.eigvals(res.W[:, 2:])))
        print "Time elapsed: %s" % str(toc-tic)
        tic=time.time()
    if ((iterations>0) and (iterations % dump_cycle_time == 0)):
        pickle.dump({"res": res, "r_hist": np.array(r_hist), "r_hist_mean": np.array(r_hist_mean), "W0": W0, "S": S,
                     "input_hist": np.array(input_hist), "inputs": inputs, "best_W": best_W, "outputs": outputs,
                     "max_reward": max_reward, "sel_probs": np.array(sel_probs)},
                    open(filebase+"%d.pkl" % iterations, 'wb'))



