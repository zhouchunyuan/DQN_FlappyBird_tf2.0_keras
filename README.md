# DQN_FlappyBird_tf2.0_keras
This is a keras version of https://github.com/yenchenlin/DeepLearningFlappyBird
The original code is written with tensorflow v1. It can work with tensorflow2.0 by a simple modification:

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

In order to confirm my understanding of DQN thieory, as well as to learn new things, I translated it into tensorflow2.0 with keras.
The code passed the test on Dell Precision 7540 laptop workstation with RTX3000. After about 20 hours training, the trained NN becomes good enough to outperform a human like me. 

1) I rewrote the DQN by keras sequence model, which is much more compact and easier understood.
2) keyboard control added, so that manual play/training is available.
3) As there exists the problem of memory leakage from tensorflow.keras.predict(), I added the following code into each loop:
        K.clear_session()
        gc.collect()
4) before predict/fitting, format change of data is needed:
            state_batch = np.array(s_j_batch).astype('float32').reshape(BATCH,80,80,4)
            target = model.predict(state_batch)
            next_state_batch = np.array(s_j1_batch).astype('float32').reshape(BATCH,80,80,4)
            readout_j1_batch = model.predict(next_state_batch)
5) the trained weights is saved in test.h5.
 
