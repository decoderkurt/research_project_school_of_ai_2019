import gym
from gym import spaces, logger
from gym.spaces import Discrete
from gym.spaces import seed
from gym.utils import seeding
import numpy as np
from PIL import Image
#from scipy.misc import imsave

from os.path import abspath, join, dirname

"""
***************
* observation *
***************
0 - No guess yet
1 - Guess is lower than the target
2 - Guess is equal to the target
3 - Guess is higher than the target

*************
* Data file *
*************
012.bmp : Blood cell 10x10 Image
012.png : Blood cell 10x10 mask

012Ori.bmp : Blood cell 300x300 Image (full size)
012Ori.png : Blood cell 300x300 mask (full size)

065_10x10.bmp : Blood cell 10x10 Image
065_10x10.png : Blood cell 10x10 mask

065_100x100.bmp : Blood cell 100x100 Image
065_100x100.png : Blood cell 100x100 mask

065Ori.bmp : Blood cell 300x300 Image (full size)
065Ori.png : Blood cell 300x300 mask (full size)
"""
TRAIN_IMAGE = "012.bmp"
TRAIN_MASK = "012.png"

class OneShotGoEnv(gym.Env):
    def __init__(self):
        seed()
        self.seed()

        script_dir = dirname(__file__)
        self.img = Image.open(abspath(join(script_dir, "..", "data", TRAIN_IMAGE)))
        self.mask = np.asarray(Image.open(abspath(join(script_dir, "..", "data", TRAIN_MASK))))
        self.mask_zero_count = np.count_nonzero(self.mask[...,0]==0)

        self.width, self.height = self.img.size

        self.action_space = spaces.Discrete(256)
        self.observation_space = spaces.Discrete(4)

        self.img_array = np.asarray(self.img.convert('L'))

        self.guess_count = 0
        self.guess_max = 100
        self.observation = 0

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #print("#######",self.guess_count)
        #print(action)
        newMaskArray = np.full([self.width, self.height], -1)
        count=  np.array([0,0])
        for i in range(self.width):
            for j in range(self.height):
                if self.img_array[i][j] >= action:
                    newMaskArray[i][j] = 0
                    count[1] += 1
                else:
                    newMaskArray[i][j] = 255
                    count[0] += 1

        if count[0] <  self.mask_zero_count:
            self.observation = 1
        elif count[0] == self.mask_zero_count:
            self.observation = 2
        else :
            self.observation = 3

        #imsave(r"E:\gym\gym\envs\pap\result2.png", newMaskArray)
        reward = ( min(count[0], self.mask_zero_count) / max(count[0], self.mask_zero_count)) ** 2
        #print(reward)

        self.guess_count += 1
        done = self.guess_count >= self.guess_max

        return self.observation, reward, done, {"mask_zero_count": self.mask_zero_count, "guesses": self.guess_count}

    def reset(self):
		print("IMAGE: ", TRAIN_IMAGE, "MASK: ", TRAIN_MASK)
        self.guess_count = 0
        self.observation = 0
        return self.observation
