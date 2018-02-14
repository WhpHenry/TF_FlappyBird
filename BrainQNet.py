
import tensorflow as tf 
import numpy as np 
import cv2
import random
from collections import deque

_FRAME_PER_ACTION = 1
_GAMMA = 0.99            # decay rate of past observations
_BATCH_SIZE = 32         # size of minibatch
_OBSERVE = 100.          # timesteps to observe before training
_EXPLORE = 150000.       # frames over which to anneal epsilon
_FINAL_EPSILON = 0.001   # final value of epsilon
_INITIAL_EPSILON = 0.01  # starting value of epsilon
_REPLAY_MEMORY = 50000   # number of previous transitions to remember
_SAVE_NET_TIMES = 1000


class BrainQNet:

    def __init__(self, action):
        # init reply memory
        self.replayMemory = deque()
        self.timeStep = 0
        self.epsilon = _INITIAL_EPSILON        
        self.action = action
        self.xInput = tf.placeholder(tf.float32, [None, 80, 80, 4])
        self.yInput = tf.placeholder(tf.float32, [None])
        self.actionInput = tf.placeholder(tf.float32, [None, self.action])
        # init Q learning network
        self.createQNetwork()

    def createQNetwork(self):
        bcr = BuildCnn()

        W_conv1 = bcr.weight_variable([8, 8, 4, 32])
        b_conv2 = bcr.bias_variable([64])

        W_conv2 = bcr.weight_variable([4, 4, 32, 64])
        b_conv1 = bcr.bias_variable([32])

        W_conv3 = bcr.weight_variable([3, 3, 64, 64])
        b_conv3 = bcr.bias_variable([64])

        W_fc1 = bcr.weight_variable([1600, 512])
        b_fc1 = bcr.bias_variable([512])

        W_fc2 = bcr.weight_variable([512, self.action])
        b_fc2 = bcr.bias_variable([self.action])

        conv1 = tf.nn.relu(bcr.conv(self.xInput, W_conv1, 4) + b_conv1)
        pool1 = bcr.max_pool_2x2(conv1)
        conv2 = tf.nn.relu(bcr.conv(pool1, W_conv2, 2) + b_conv2)
        # pool2 = bcr.max_pool_2x2(conv2)
        conv3 = tf.nn.relu(bcr.conv(conv2, W_conv3, 1) + b_conv3)
        # pool3 = bcr.max_pool_2x2(conv3)
        conv3_flat = tf.reshape(conv3, [-1, 1600])
        fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)

        # Q learning layer
        self.Qvalue = tf.matmul(fc1, W_fc2) + b_fc2
        # self.getAction = tf.placeholder(tf.float32, [None, self.action])
        # self.getY = tf.placeholder(tf.float32, [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Qvalue, self.actionInput), reduction_indices = 1)
        cost = tf.reduce_mean(tf.square(self.yInput - Q_action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(cost)

        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:                 
            # load trained network
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print ("Could not find old network weights")

    def trainQNetwork(self, reward, oriImg):
        minibatch = random.sample(self.replayMemory, _BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # calculate y
        y_batch = []
        Qvalue_batch = self.Qvalue.eval(feed_dict={self.xInput: nextState_batch})
        for i in range(0, _BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + _GAMMA * np.max(Qvalue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.xInput: state_batch
            })


        if self.timeStep % 100 == 0:

            print("** TimeStep: {}, Reward: {}, Epsilon: {}, **".format(self.timeStep, reward, self.epsilon))

        if self.timeStep % _SAVE_NET_TIMES == 0 :
            print("In {} time step training, Saving session".format(self.timeStep))
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)
            cv2.imwrite("saved_networks/" + str(self.timeStep) + ".png", oriImg)

        # if self.timeStep > _MAX_TRAINING_TIMES:
        #     raise Exception("over max training times")

    def setPerception(self, nextImg, action, reward, terminal, oriImg):
        newState = np.append(self.currentState[:,:,1:],nextImg,axis = 2)
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))
        if len(self.replayMemory) > _REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > _OBSERVE:
            # print("training in: ", self.timeStep)
            self.trainQNetwork(reward, oriImg)
        self.currentState = newState
        self.timeStep += 1


    def getAction(self):
        Qvalue = self.Qvalue.eval(feed_dict={self.xInput:[self.currentState]})[0]
        action = np.zeros(self.action)
        action_idx = 0
        if self.timeStep % _FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_idx = random.randrange(self.action)
                action[action_idx] = 1
            else:
                action_idx = np.argmax(Qvalue)
                action[action_idx] = 1
        else:
            action[0] = 1 # do nothing

        if self.epsilon > _FINAL_EPSILON and self.timeStep > _OBSERVE:
            self.epsilon -= (_INITIAL_EPSILON - _FINAL_EPSILON)/_EXPLORE

        return action

    def setInitState(self, gameImg0):
        self.currentState = np.stack((gameImg0, gameImg0, gameImg0, gameImg0), axis = 2)


class BuildCnn:
    def __init__(self, sdev = 0.01, bias=0.1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = "SAME"):
        self.sdev = sdev
        self.bias = bias
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def weight_variable(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=self.sdev))

    def bias_variable(self, shape):
        return tf.Variable(tf.constant(self.bias, shape=shape))

    def conv(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=self.padding)

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)
