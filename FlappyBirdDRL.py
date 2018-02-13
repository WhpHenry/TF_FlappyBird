import cv2
import numpy as np
import BrainQNet as bqnet

import sys
sys.path.append('game/')
import wrapped_flappy_bird as game

def preprocess(gameImg):
    # first layer of cnn need 80 * 80 img as input
    # tranlate color space from RGB to GRAY
    gameImg = cv2.cvtColor(cv2.resize(gameImg, (80, 80)), cv2.COLOR_BGR2GRAY)   
    # binary img color
    _, gameImg = cv2.threshold(gameImg, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(gameImg, (80, 80, 1))

def playFlappyBird():
    action = 2
    instance_brain = bqnet.BrainQNet(action)
    instance_game = game.GameState()
    # set init state
    action0 = np.array([1,0])   # [1,0] do nothing; [0,1] flap the bird
    gameImg0, reward0, terminal  = instance_game.frame_step(action0)
    gameImg0 = cv2.cvtColor(cv2.resize(gameImg0, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, gameImg0 = cv2.threshold(gameImg0, 1, 255, cv2.THRESH_BINARY)
    instance_brain.setInitState(gameImg0)

    # start game
    while True:
        action = instance_brain.getAction()
        nextImg, reward, terminal = instance_game.frame_step(action)
        nextImg = preprocess(nextImg)
        instance_brain.setPerception(nextImg, action, reward, terminal)


def test():
    playFlappyBird()

if __name__ == '__main__':
    test()