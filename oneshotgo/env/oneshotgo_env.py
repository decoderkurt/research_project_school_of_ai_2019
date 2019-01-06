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
0 - No guess yet submitted (only after reset)
1 - Guess is lower than the target
2 - Guess is equal to the target
3 - Guess is higher than the target
"""
class OneShotGoEnv(gym.Env):
    def __init__(self):
        seed()
        self.seed()

        script_dir = dirname(__file__)
        self.img = Image.open(abspath(join(script_dir, "..", "data", "012.bmp")))
        #self.img = Image.open(r"E:\gym\gym\envs\pap\065.bmp")
        #self.img = Image.open(r"E:\gym\gym\envs\pap\012Ori.bmp")
        self.mask = np.asarray(Image.open(abspath(join(script_dir, "..", "data", "012.png"))))
        #self.mask = np.asarray(Image.open(r"E:\gym\gym\envs\pap\065.png"))
        #self.mask = np.asarray(Image.open(r"E:\gym\gym\envs\pap\012Ori.png"))
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
        self.guess_count = 0
        self.observation = 0
        return self.observation
