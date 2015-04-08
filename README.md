# HBP_D4.6.3
Material for deliverable D4.6.3 of the Human Brain Project:

The code in this repository corresponds to the experiment described in section 3.2.2 of the accompanying paper “Reward modulated Hebbian plasticity as leverage for partially embodied
control in compliant robotics”, submitted to Frontiers in Neurorobotics. It demonstrates the application of reward modulated Hebbian learning with delayed rewards in a setting that emulates embodied computation. The task consists of the classification of 8 different analog signals, corresponding to encoded bit patterns (see section 3.2.2 of the paper for a full description). The input signals drive a neural network, half of which model a ‘brain’ that can be adapted by learning. The other half is fixed and acts as a dummy for a body. The output is generated from signals in the ‘body’.  The connection weights in both, the body and the brain are randomly initialised. 

