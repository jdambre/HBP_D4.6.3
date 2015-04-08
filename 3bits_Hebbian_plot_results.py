__author__ = 'jdambre'
import numpy as np
import pickle
from matplotlib import pylab as plt
import os

last_file_only=True
iteration_to_plot="1000000"
do_all=False
filepath="res_3bits_andrea/"
filebase="res_3bits_andrea_"
baselen=len(filebase)

if last_file_only:
    iteration_to_plot=0
    files=os.listdir("./"+filepath)
    numfiles=len(files)
    for i in range(numfiles):
        if files[i][:baselen]==filebase:
            iteration_to_plot=max(iteration_to_plot,int(files[i][baselen:-4]))
    print "plotting last dump, recorded at "+str(iteration_to_plot)+" iterations \n"

plt.ion()
#r = pickle.load(open("res_3bits_andrea_153603.pkl"))
r = pickle.load(open(filepath+filebase+str(iteration_to_plot)+".pkl"))

inputs = r["inputs"]
res = r["res"]
#res2 = r2["res"]
inputs_string = ["001", "100", "101", "000", "011", "111", "010", "110"]
#targets = [-1,1,1,-1]
targets = r["outputs"]  #outputs.copy()
reward_history=r["r_hist"]


print "plotting figure 0 (reward evolution)"
plt.figure()
plt.subplot(2,1,1)
plt.plot(range(reward_history.shape[0]),10**reward_history)
plt.title('Evolution of MSE (Reward = -MSE)')
plt.xlabel('Iteration')
plt.ylabel('10**(- MSE)')
plt.subplot(2,1,2)
plt.plot(range(reward_history.shape[0]),np.log(-reward_history))
plt.xlabel('Iteration')
plt.ylabel('log(MSE)')
plt.show()

if do_all:
    print "plotting figure 1"
    plt.figure()
    for i in xrange(8):
        plt.subplot(2, 4, i + 1)
        d = inputs[i:i + 1, :].T
        states = res.run(d, 0)[-1]
        plt.plot([0, 0], [-1.5, 1.5], 'w')

        plt.plot(d.ravel(), '+', label='input')
        plt.plot(np.arange(25, 30), np.ones(5) * targets[i], '.', label='target')
        plt.plot(states[:, [42, 59]].sum(1), label='output')

        plt.title("%s - %f" % ( inputs_string[i], np.round(targets[i], 2)))
    plt.suptitle('Reservoir test no noise - forward order')
    plt.legend()
    plt.show()

    print "plotting figure 2"
    plt.figure()
    for i in xrange(7, -1, -1):
        plt.subplot(2, 4, i + 1)
        d = inputs[i:i + 1, :].T
        states = res.run(d, 0)[-1]
        plt.plot([0, 0], [-1.5, 1.5], 'w')

        plt.plot(d.ravel(), '+', label='input')
        plt.plot(np.arange(25, 30), np.ones(5) * targets[i], '.', label='target')
        plt.plot(states[:, [42, 59]].sum(1), label='output')

        plt.title("%s - %f" % ( inputs_string[i], np.round(targets[i], 2)))

    plt.suptitle('Reservoir test no noise - reversed order')
    plt.subplot(2, 4, 8)
    plt.legend()
    plt.show()

    print "Running data for figure 3"
    trials_per_input = 50
    results_per_input = np.zeros(8).astype(int)
    results_all = np.zeros((8, 30, 100, trials_per_input))
    while True:
        goal = np.random.randint(8)
        if (results_per_input[goal] == trials_per_input):
            continue

        d = inputs[goal:goal + 1, :].T
        states = res.run(d, 0)[-1]
        results_all[goal, :, :, results_per_input[goal]] = states
        results_per_input[goal] += 1
        if (results_per_input.sum() == trials_per_input * 8):
            break

    print "plotting figure 3"
    plt.figure()
    for i in xrange(8):
        plt.subplot(2, 4, i + 1)
        plt.plot([0, 0], [-1.5, 1.5], 'w')
        states = results_all[i]
        d = inputs[i:i + 1, :].T
        plt.plot(d.ravel(), '+', label='input')

        for j in xrange(trials_per_input):
            if (j > 0):
                plt.plot(states[:, [42, 59], j].sum(1), 'black')
            else:
                plt.plot(states[:, [42, 59], j].sum(1), 'black', label='outputs')
        plt.plot(np.arange(25, 30), np.ones(5) * targets[i], '.', label='target')

        plt.title("%s: %f" % ( inputs_string[i], targets[i]))

    plt.suptitle('Reservoir test no noise - %d different orders' % trials_per_input)
    plt.legend(loc=4)
    plt.show()


    print "Running data for figures 4 and 5"
    trials_per_input = 50
    noise_levels = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    num_noise_levels = noise_levels.shape[0]
    results_per_input = np.zeros((8, num_noise_levels)).astype(int)
    results_all = np.zeros((8, 30, 100, trials_per_input, num_noise_levels))
    inputs_all = np.zeros((8, 30, 1, trials_per_input, num_noise_levels))
    for k in xrange(num_noise_levels):
        while True:
            goal = np.random.randint(8)
            if (results_per_input[goal, k] == trials_per_input):
                continue

            d = inputs[goal:goal + 1, :].T.copy()
            d += np.random.randn(d.shape[0], d.shape[1]) * noise_levels[k]
            inputs_all[goal, :, :, results_per_input[goal, k], k] = d
            states = res.run(d, 0)[-1]
            results_all[goal, :, :, results_per_input[goal, k], k] = states
            results_per_input[goal, k] += 1
            if (results_per_input[:, k].sum() == trials_per_input * 8):
                break

    print "plotting figure 4"
    plt.figure()
    for i in xrange(8):
        plt.subplot(2, 4, i + 1)

        for k in xrange(num_noise_levels - 1, -1, -1):
            states = results_all[i, :, :, :, k]
            col = str(np.linspace(0.2, .8, 9)[k])
            d = inputs_all[i, :, 0, :, k]
            for j in xrange(trials_per_input):
                plt.plot(states[:, [42, 59], j].sum(1), col)

        plt.plot(np.arange(25, 30), np.ones(5) * targets[i], '.', label='target')
        plt.ylim(-1.2, 1.2)
        plt.title("%s: %f" % ( inputs_string[i], targets[i]))

    plt.suptitle('Reservoir input noise - %d different orders' % trials_per_input)
    plt.legend(loc=4)
    plt.show()

    print "plotting figure 5"
    plt.figure()
    for i in xrange(8):
        plt.subplot(2, 4, i + 1)

        for k in xrange(num_noise_levels - 1, -1, -1):
            states = results_all[i, :, :, :, k]
            col = str(np.linspace(0.2, .8, 9)[k])
            d = inputs_all[i, :, 0, :, k]
            for j in xrange(trials_per_input):
                plt.plot(d[:, j], col)

        plt.ylim(-1.2, 1.2)
        plt.title("%s: %f" % ( inputs_string[i], targets[i]))

    plt.suptitle('Reservoir input noise INPUTS - %d different orders' % trials_per_input)
    plt.legend(loc=4)
    plt.show()

    print "Running data for figures 6 and 7"
    trials_per_input = 50
    noise_levels = np.array([0, 0.025, 0.05, 0.1])
    num_noise_levels = noise_levels.shape[0]
    results_per_input = np.zeros((8, num_noise_levels)).astype(int)
    results_all = np.zeros((8, 30, 100, trials_per_input, num_noise_levels))
    for k in xrange(num_noise_levels):
        while True:
            goal = np.random.randint(8)
            if (results_per_input[goal, k] == trials_per_input):
                continue

            d = inputs[goal:goal + 1, :].T.copy()
            states = res.run(d, noise_levels[k])[-1]
            results_all[goal, :, :, results_per_input[goal, k], k] = states
            results_per_input[goal, k] += 1
            if (results_per_input[:, k].sum() == trials_per_input * 8):
                break

    print "plotting figure 6"
    plt.figure()
    for i in xrange(8):
        plt.subplot(2, 4, i + 1)

        d = inputs[i:i + 1, :].T
        plt.plot(d.ravel(), '+', label='input')
        for k in xrange(num_noise_levels - 1, -1, -1):
            states = results_all[i, :, :, :, k]
            col = str(np.linspace(0.2, .8, 9)[k])
            for j in xrange(trials_per_input):
                plt.plot(states[:, [42, 59], j].sum(1), col)

        plt.plot(np.arange(25, 30), np.ones(5) * targets[i], '.', label='target')
        plt.ylim(-1.2, 1.2)
        plt.title("%s: %f" % ( inputs_string[i], targets[i]))

    plt.suptitle('Reservoir state noise 0->0.1 - %d different orders' % trials_per_input)
    plt.legend(loc=4)
    plt.show()

    print "plotting figure 7"
    plt.figure()
    for i in xrange(8):
        plt.subplot(2, 4, i + 1)

        d = inputs[i:i + 1, :].T
        plt.plot(d.ravel(), '+', label='input')
        k = 0
        states = results_all[i, :, :, :, k]
        for j in xrange(trials_per_input):
            if (j == 0):
                plt.plot(states[:, 42, j], 'r', label='neuron 1')
                plt.plot(states[:, 59, j], 'g', label='neuron 2')
                plt.plot(states[:, [42, 59], j].sum(1), 'b', label='output (sum)')
            else:
                plt.plot(states[:, 42, j], 'r')
                plt.plot(states[:, 59, j], 'g')
                plt.plot(states[:, [42, 59], j].sum(1), 'b')

        plt.plot(np.arange(25, 30), np.ones(5) * targets[i], '.', label='target')
        plt.ylim(-1.2, 1.2)
        plt.title("%s: %f" % ( inputs_string[i], targets[i]))

    plt.suptitle('output neurons (before sum) no noise- %d different orders' % trials_per_input)
    plt.legend(loc=4)


    # trials_per_input = 1000
    # noise_levels = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    # num_noise_levels = noise_levels.shape[0]
    # results_per_input = np.zeros((8, num_noise_levels)).astype(int)
    # results_all = np.zeros((8, 30, 100, trials_per_input, num_noise_levels), dtype=np.float32)
    # inputs_all = np.zeros((8, 30, 1, trials_per_input, num_noise_levels), dtype=np.float32)
    # for k in xrange(num_noise_levels):
    #     print k
    #     while True:
    #         goal = np.random.randint(8)
    #         if (results_per_input[goal, k] == trials_per_input):
    #             continue
    #
    #         d = inputs[goal:goal + 1, :].T.copy()
    #         d += np.random.randn(d.shape[0], d.shape[1]) * noise_levels[k]
    #         inputs_all[goal, :, :, results_per_input[goal, k], k] = d
    #         states = res.run(d, 0)[-1]
    #         results_all[goal, :, :, results_per_input[goal, k], k] = states
    #         results_per_input[goal, k] += 1
    #         if (results_per_input[:, k].sum() == trials_per_input * 8):
    #             break
    #
    #

plt.ioff()
plt.show()