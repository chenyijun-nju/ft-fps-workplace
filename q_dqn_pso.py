import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
from tensorflow.keras import models, layers, optimizers

#---------------------------------DQN---------------------------------------
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
            layers.Dense(50, input_dim=STATE_DIM, activation='relu'),
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
    
class qlearning():
    def __init__(self):
        self.actions = [0, 1, 2]
        self.learning_rate = 0.01
        self.Q = np.zeros((2,3))
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def Qfresh(self, state, action, next_state, reward):
        current_q = self.Q[state][action]
        new_q = reward + self.discount_factor * max(self.Q[next_state])
        self.Q[state][action] += self.learning_rate * (new_q - current_q)
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 贪婪策略随机探索动作
            action = np.random.choice(self.actions)
        else:
            # 从q表中选择
            state_action = self.Q[state]
            action = self.arg_max(state_action)
        return action

    def arg_max(self, state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

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
        self.pbeststate = np.zeros((self.pN, self.dim))# 存储上一周期的pbest
        self.gbeststate = np.zeros((1, self.dim))
        self.checkperiod = 2# 收敛状态检查周期
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
    
    def statememory(self):
        self.pbeststate = self.pbest.copy()
        self.gbeststate = self.gbest.copy()

    def statecheck(self):
        time = 0 
        for i in range(self.pN):
            if (self.pbest[i] - self.pbeststate[i]).any() == False:
                time += 1
        if time >= 5:
            return 1
        else:
            return 0

    def action_do(self, action):
        if action == 0:
            #print('before moving gbest:', self.gbest)
            for i in range(self.pN):
                for j in range(self.dim):
                    if self.X[i][j] != self.pbest[i][j]:
                        d = self.X[i][j] - self.pbest[i][j]
                        single = d/abs(d)
                        self.X[i][j] += single * random.uniform(0, 5.12 - abs(self.pbest[i][j]))
            #print('before moving gbest:', self.gbest)
        elif action == 1:
            for i in range(self.pN):
                for j in range(self.dim):
                    if self.X[i][j] != self.pbest[i][j]:
                        d = self.X[i][j] - self.pbest[i][j]
                        single = d/abs(d)
                        self.X[i][j] += -single * random.uniform(0, d)
    
    def rewardcheck(self):
        if (self.gbest - self.gbeststate).any():
            return 50
        else:
            return 0
        
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
            return 2
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
                self.V[i][j] = random.uniform(0, 5.12)
            self.pbest[i] = self.X[i].copy()
            tmp = self.function(self.X[i])
            self.dqn_v = random.uniform(-0.1, 0.1)
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gbest = self.X[i].copy
            self.statememory()
 
                # ----------------------更新粒子位置----------------------------------
 
    def iterator(self):
        fitness = []
        titi = 0
        QL = qlearning()
        agent = DQN()
        flag = 1
        for t in range(self.max_iter):
            titi += 1
            self.pbestbefore = self.pbest.copy()
            self.gbestbefore = self.gbest.copy()
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.X[i])
                if temp < self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i].copy()
                    if self.p_fit[i] < self.fit:
                        print('new gbest found')
                        self.gbest = self.X[i].copy()
                        print('gbest now:', self.gbest)
                        self.fit = self.p_fit[i]
            if titi == self.checkperiod:
                titi = 0
                state = self.statecheck()
                action_q = QL.get_action(state)
                #print('before moving gbest:', self.gbest)
                self.action_do(action_q)
                #print('after moving gbest:', self.gbest)
                flag = 0
            if titi == self.checkperiod-1 and flag == 0:
                reward = self.rewardcheck()
                next_state = self.statecheck()
                self.statememory()
                QL.Qfresh(state,action_q,next_state,reward)
            for i in range(self.pN):
                #print('before moving gbest:', self.gbest)
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
                dqn_re = self.dqn_reward(i)
                agent.remember(self.dqn_v, dqn_act, dqn_next_state, dqn_re)
                agent.train()
                self.dqn_v = dqn_next_state
                self.Xbefore[i] = self.X[i].copy()
                #print('after moving gbest:', self.gbest)
            fitness.append(self.fit)
            print('time:',t, end=' ')
            print(self.gbest, end=" ")
            print(self.fit)  # 输出最优值
            #print(QL.Q)
        return fitness
 
        # ----------------------程序执行-----------------------
 
 
my_pso = PSO(pN=30, dim=4, max_iter=200)
my_pso.init_Population()
fitness = my_pso.iterator()
# -------------------画图--------------------
plt.figure(1)
plt.title("Q-DQN-pso")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, 200)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()