# About

I tried to solve real world problems with machine learning in this research project. I noted that there is a limit to the traditional Deep Learning application, which is highly dependent on existing datasets because it is very difficult to obtain labled data in the biomedical sector.

For applying to biomedical field, the basis for judgment must be clear, so I decided to use image among various types of data for the reason of being visualized intuitively using mask.

After training only one labeled image data, I wanted to categorize a lot of unseen data based on it, and to solve the basic concept of one-shot learning through reinforcement learning.

In this project, I redefined the one-shot image segmentation problem as a reinforcement learning and solved it using PPO. I found that there was actually a dramatic performance.


# Reinforcement learning

I defined the human's ability to read images as a policy of reinforcement learning, and an agent's prediction of this as an action. I also considered inverse reinforcement learning and GAIL. In this case, the reward function is pretty clear and the policy can be important. I chose PPO that also does not need the MDP(Markov Descion Proecess).

I used PPO of OpenAI gym, and implemented custom env for this project. I felt a similarity with GO in that an agent creates a grayscale mask from the original RGB image, so named it as "OneShotGo".

# Reward Function
The agent reads the original image and converts it into a two-dimensional array as large as the image size, and performs a back-white calibration by comparing the pixel value with the predicted value. I designed the reward function with the correct rate compared to the actual labled mask. 

In other words, the agent produces a mask every time through repeated actions, which will receive a higher reward if they are similar to the correct answer.
```
reward = ( min(count[0], self.mask_zero_count) / max(count[0], self.mask_zero_count)) ** 2
```
The key to this return function is using the min max function so that the number of zeros is the most important and the correctness, whether large or small, is equally affected. Given the nature of biomedical images, background and object classification are most important, and slide images are mostly colored, so the better the background is blown away, the higher the reward.

I also considered using MSE and SSIM, but the former was not appropriate due to high variance and the latter was always highly similarity.


python -m baselines.run --alg=ppo2 --env=OneShotGo-v0
python -m baselines.run --alg=ppo2 --env=OneShotGo-v0 --load_path="OneShotGo10M"
