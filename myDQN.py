import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
from tensorflow.keras import models, layers, optimizers
# --------------------DQN------------------------------------------
class DQN(object):
    def __init__(self):
        self.step = 0
        self.state_dim = 1
        self.action_dim = 3
        self.update_freq = 50  # 模型更新频率
        self.replay_size = 100  # 训练集大小
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.epsilon = 0.2

    def create_model(self):
        STATE_DIM = self.state_dim
        ACTION_DIM = self.action_dim
        model = models.Sequential([
            layers.Dense(100, input_dim=STATE_DIM, activation='relu'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model
    
    def act(self, s):
        epsilon = self.epsilon
        if np.random.uniform() < epsilon:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.array([s]))[0])

    def save_model(self, file_path='mydqn.h5'):
        print('model saved')
        self.model.save(file_path)

    def remember(self, s, a, next_s, reward):
        self.replay_queue.append((s, a, next_s, reward))

    def train(self, batch_size=50, lr=0.85, factor=0.95):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        # 使用公式更新训练集中的Q值
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))
        
        # 传入网络进行训练
        self.model.fit(s_batch, Q, verbose=0)


# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, pN, dim, max_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值
        self.Xbefore = np.zeros((self.pN, self.dim))# 存储DQN上一次的坐标
        self.pbestbefore = np.zeros((self.pN, self.dim))
        self.gbestbefore = np.zeros((1, self.dim))
        self.distence = 1
        self.dqn_v = 0
 
    # ---------------------目标函数Sphere函数-----------------------------
    def function(self, X):
        igt = 0
        for i in range(self.dim):
            igt += X[i]**2 - 10 * math.cos(2 * math.pi * X[i])
        return 10 * self.dim + igt

    def dqnaction_do(self, action):
        if action == 0:
            return 0.01
        elif action == 1:
            return -0.01
        elif action == 2:
            return 1
    
    def dqn_reward(self, i):
        d = 0
        for j in range(self.dim):
            d += abs(self.X[i][j] - self.Xbefore[i][j])
        if d > self.distence:
            return 0
        elif (self.pbest[i] - self.pbestbefore[i]).any() == True:
            return 5
        elif (self.gbest - self.gbestbefore).any() == True:
            return 10
        else:
            return -5


    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(-5.12, 5.12)
                self.V[i][j] = random.uniform(0, 1)
            self.pbest[i] = self.X[i]
            self.dqn_v = random.uniform(-0.1, 0.1)
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gbest = self.X[i]
 
                # ----------------------更新粒子位置----------------------------------
 
    def iterator(self):
        fitness = []
        agent = DQN()
        for t in range(self.max_iter):
            self.pbestbefore = self.pbest
            self.gbestbefore = self.gbest
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.X[i])
                if temp < self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if self.p_fit[i] < self.fit:  # 更新全局最优
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            for i in range(self.pN):
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i]) + self.dqn_v
                self.X[i] = self.X[i] + self.V[i]
                dqn_act = agent.act(self.dqn_v)
                actionnow = self.dqnaction_do(dqn_act)
                if actionnow == 1:
                    self.dqn_v = 0
                    dqn_next_state = self.dqn_v
                else:
                    dqn_next_state = self.dqn_v + actionnow
                print("dqn action:",actionnow,'now dqn v',self.dqn_v)
                dqn_re = self.dqn_reward(i)
                agent.remember(self.dqn_v, dqn_act, dqn_next_state, dqn_re)
                agent.train()
                self.dqn_v = dqn_next_state
                self.Xbefore[i] = self.X[i]
            fitness.append(self.fit)
            print("the time:",t)
            print(self.gbest, end=" ")
            print(self.fit)  # 输出最优值
        return fitness
 
        # ----------------------程序执行-----------------------
 
 
my_pso = PSO(pN=30, dim=3, max_iter=200)
my_pso.init_Population()
fitness = my_pso.iterator()
# -------------------画图--------------------
plt.figure(1)
plt.title("dqn-pso")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, 200)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()