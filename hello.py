# Deep-Q learning Agent
import tensorflow as tf
import gym.envs as envs
import gym
import numpy as np
import tensorflow.keras as keras
#查看所有游戏列表
print(envs.registry.all())

#启动平衡车游戏
env = gym.make('CartPole-v0')
env.reset()
while True:
     env.render()
     env.action_space(1)
