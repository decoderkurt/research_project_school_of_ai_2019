from gym.envs.registration import register

register(
    id='OneShotGo-v0',
    entry_point='oneshotgo.env:OneShotGoEnv',
)