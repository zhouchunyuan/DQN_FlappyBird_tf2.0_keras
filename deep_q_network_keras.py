#!/usr/bin/env python
from __future__ import print_function
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import RandomNormal,RandomUniform
from tensorflow.keras import backend as K
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import keyboard
import gc
GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 5000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 128# size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-6 # learning rate

def createNetwork():
    model =tensorflow.keras.Sequential([
        keras.layers.Conv2D(32, (8, 8),strides=(4, 4),padding='same', activation='relu', 
                            input_shape=(80, 80, 4),kernel_initializer = RandomUniform()),
        keras.layers.MaxPooling2D((2, 2),padding='same'),
        keras.layers.Conv2D(64, (4,4),strides=(2, 2), padding='same',activation='relu',kernel_initializer = RandomUniform()),
        #keras.layers.MaxPooling2D((2, 2),padding='same'),
        keras.layers.Conv2D(64, (3,3),padding='same',activation='relu',kernel_initializer = RandomUniform()),
        #keras.layers.MaxPooling2D((2, 2),padding='same'),
        keras.layers.Flatten(),
        #keras.layers.Dense(256,activation='relu',kernel_initializer = RandomUniform()),
        keras.layers.Dense(1600,activation='relu',kernel_initializer = RandomUniform()),
        keras.layers.Dense(512, activation='relu',kernel_initializer = RandomUniform()),
        keras.layers.Dense(2, activation='relu',kernel_initializer = RandomUniform())
    ])
    model.compile(loss='mse',
                  optimizer=Adam(lr=LEARNING_RATE))
    return model

def trainNetwork(model):
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    try:
        model.load_weights('test.h5')
        print("Successfully loaded:")
    except:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
    
        # choose an action epsilon greedily
        state = s_t.astype('float32').reshape(1,80,80,4)
        readout_t = model.predict(state)
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:# and t > OBSERVE:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        #######################
        # enable manual control
        #######################
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                model.save('test.h5')
                print('model saved, You Pressed q Key!')
                break  # finishing the loop
            if keyboard.is_pressed(' '):
                a_t =np.array([0.,1.])# go up 
                print('up')
        except:
            break 
            
        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            state_batch = np.array(s_j_batch).astype('float32').reshape(BATCH,80,80,4)
            target = model.predict(state_batch)
            next_state_batch = np.array(s_j1_batch).astype('float32').reshape(BATCH,80,80,4)
            readout_j1_batch = model.predict(next_state_batch)

            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    target[i][np.argmax(a_batch[i])] = r_batch[i]
                else:
                    #y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                    target[i][np.argmax(a_batch[i])] = r_batch[i] + GAMMA * (np.max(readout_j1_batch[i]))

            # perform gradient step
            state_batch = np.array(s_j_batch).astype('float32').reshape(BATCH,80,80,4)
            model.train_on_batch(state_batch,target)

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            model.save('test.h5')

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t),\
            "/ Q_MIN %e" % np.min(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''
        K.clear_session()
        gc.collect()
def playGame():
    model = createNetwork()
    trainNetwork(model)

def main():
    playGame()

if __name__ == "__main__":
    main()
