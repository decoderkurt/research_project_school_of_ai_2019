[<img src="oneshotgo/data/res/logo.png" width=70%/>](https://www.theschool.ai/school-of-ai-fellowship/)

# One shot learning using Proximal Policy Optimization 
**Kurt Koo** callmekoo@gmail.com  *Research Fellow, School of AI*  [![linkedin](oneshotgo/data/res/linkedin.png)](https://www.linkedin.com/in/kurtkoo)   [![twitter](oneshotgo/data/res/twitter.png)](https://twitter.com/kurt_koo)  [![facebook](oneshotgo/data/res/fb.png)](https://www.facebook.com/vcmlrl)


## Introduction
In this research project, to solve real world problems with machine learning, I noted that there is a limit to the traditional Deep Learning application, which is highly dependent on existing datasets because it is still difficult to obtain enough labled data.

The basis for judgment must be clear in the biomedical field, so I decided to use image data among various types for the reason of being visualized intuitively.

Using just one labeled image data for training, I wanted to categorize a lot of unseen data based on it by the basic concept of one shot learning through reinforcement learning.

In this project, I redefined the one shot image segmentation problem as a reinforcement learning and solved it using PPO. I found that there was actually a dramatic performance.


## Reinforcement learning
<p align="center">
<img src="oneshotgo/data/res/un.png" width=70%/>
</p>
I defined the human's ability to read images as a policy of reinforcement learning, and an agent's prediction of this as an action. I also considered inverse reinforcement learning and GAIL. But, in this case, the reward function is pretty clear and the policy is more important, I descided to use PPO that also does not need the MDP(Markov Decision Process).

I used PPO of OpenAI gym, and implemented custom env for this project. I felt a similarity with GO in that an agent creates a grayscale mask from the original RGB image, so named it as "OneShotGo".

### Reward Function
The agent reads the original image and converts it into a two-dimensional array as large as the image size, and performs a black-white calibration by comparing the pixel value with the predicted value. I designed the reward function with the correct response rate compared to the actual labled mask. 

In other words, the agent produces a mask every time through repeated actions, which will receive a higher reward if they are similar to the correct answer.
```
reward = ( min(count[0], self.mask_zero_count) / max(count[0], self.mask_zero_count)) ** 2
```
The key to this reward function is using the min max function so that the number of zeros is the most important and the correctness, whether large or small, is equally affected. Given the nature of biomedical images, background and object classification is the most important, and slide images are mostly colored, so the better the background is blown away, the higher the reward.

I also considered using MSE and SSIM, but the former was not appropriate due to high variance and the latter was always highly similarity.

### Action
The intention was to distinguish the background from the cell boundary and the nucleus at once with the black, grey and white. To do this, two separate uint8 between 0 and 255 are required for action_space. There is still a problem where Tuple action_space is not implemented yet in PPO of OpenAI, and in the case of Box, a bug with an action value of between -1.0 and 1.0 was found as float, regardless of defined the action space. I eventually used only one discrete integer.

### action_space, observation_space
Discrete or -1.0 to 1.0 Box action_space, are already widely used in games such as Arati and seem to be well implemented. It works well  with observation_space, not action_space. Until the fix, it would be better to be careful if you apply PPO of gym in a unique way.

### keras-rl, tensorforce, ray, SLM
keras-rl has not yet implemented a PPO. In case of tensorforce, it was unstable because it did not fit my development environment. Ray does not yet support for Windows. In the case of SLM, the dependency of the ray makes it not support for Windows. I installed and tested Linux in Windows using WSL, but due to the instability of WSL, the system was failed while apt-get update. OpenAI was my best choice.

## Experiment
### Install
```
git clone https://github.com/decoderkurt/research_project_school_of_ai_2019.git
cd research_project_school_of_ai_2019
pip install -e .
```
### Train
```
python -m baselines.run --alg=ppo2 --env=OneShotGo-v0 --save_path="YourOwnOneShotGo10M"
```
### Test
```
python -m baselines.run --alg=ppo2 --env=OneShotGo-v0 --load_path="OneShotGo10M"
```
## Result
### Train
#### 10x10 image (012.bmp)

<img src="oneshotgo/data/012.bmp" width=30% /><img src="oneshotgo/data/012.png" width=30%  />

|Before|After|
|---|---|
|<img src="oneshotgo/data/result/012_10x10_not_trained.PNG" />|<img src="oneshotgo/data/result/012_10x10_trained.PNG" />|

### Test
#### 10x10 unseen image (065.bmp)
<img src="oneshotgo/data/065_10x10.bmp" width=30%/><img src="oneshotgo/data/065_10x10.png" width=30% />

|Before|After|
|---|---|
|<img src="oneshotgo/data/result/065_10x10_unseen_oneshot_without_oneshot.PNG"  />|<img src="oneshotgo/data/result/065_10x10_unseen_oneshot.PNG" />|

#### 100x100 unseen image (065.bmp)
<img src="oneshotgo/data/065_100x100.bmp" width=30%/><img src="oneshotgo/data/065_100x100.png" width=30% />

|Before|After|
|---|---|
|<img src="oneshotgo/data/result/065_100x100_unseen_oneshot_without_oneshot.PNG" />|<img src="oneshotgo/data/result/065_100x100_unseen_oneshot.PNG"  />|

## Conclusion
||Training(012_10x10)|Test(012_10x10)|Test(065_10x10)|
|---|---|---|---|
|Result|<img src="oneshotgo/data/res/plot.PNG" />|<img src="oneshotgo/data/res/plot2.PNG" />|<img src="oneshotgo/data/res/plot3.PNG" />|
|Timesteps|1e6|1e6|1e6|

|Filename|Size(pixel)|Before|After|Effect|
|---|---|---|---|---|
|012.bmp|10x10|9.05|64.3|710%↑|
|065.bmp|10x10|22.3|77.6|347%↑|
|065.bmp|100x100|17.9|61.3|342%↑|

Using only one image PPO training, I got about three times more effective improvement than if it did not apply. Through this research project, I saw the possibility of solving real world problems using reinforcement learning where traditional deep learning could not be applied due to small data.

Also, I can see the reinforcement learning outcomes using PPO worked well even in different size unseen images. I think the strength of reinforcement learning is that it can be applied to more complex and time-consuming data after learning it quickly with a small size data.

## Colab link
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decoderkurt/research_project_school_of_ai_2019/blob/master/Research_Project_SchoolofAI.ipynb) 
https://colab.research.google.com/github/decoderkurt/research_project_school_of_ai_2019/blob/master/Research_Project_SchoolofAI.ipynb

## Reference
[1]OpenAI https://github.com/openai/baselines https://github.com/openai/gym <br>
[2]Proximal Policy Optimization Algorithms https://arxiv.org/abs/1707.06347 <br>
[3]Dataset https://github.com/zxaoyou/segmentation_WBC
