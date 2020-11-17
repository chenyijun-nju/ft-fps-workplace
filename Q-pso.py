# coding: utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import math
 
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
        self.pbeststate = np.zeros((self.pN, self.dim))# 存储上一周期的pbest
        self.gbeststate = np.zeros((1, self.dim))
        self.checkperiod = 20# 收敛状态检查周期
 
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
            for i in range(self.pN):
                for j in range(self.dim):
                    if self.X[i][j] != self.pbest[i][j]:
                        d = self.X[i][j] - self.pbest[i][j]
                        single = d/abs(d)
                        self.X[i][j] += single * random.uniform(0, 5.12 - abs(self.pbest[i][j]))
        elif action == 1:
            for i in range(self.pN):
                for j in range(self.dim):
                    if self.X[i][j] != self.pbest[i][j]:
                        d = self.X[i][j] - self.pbest[i][j]
                        single = d/abs(d)
                        self.X[i][j] += -single * random.uniform(0, d)
    
    def rewardcheck(self):
        if (self.gbest - self.gbeststate).any():
            return 20
        else:
            return 0

    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(-5.12, 5.12)
                self.V[i][j] = random.uniform(0, 5.12)
            self.pbest[i] = self.X[i].copy()
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gbest = self.X[i].copy()
            self.statememory()
 
                # ----------------------更新粒子位置----------------------------------
 
    def iterator(self):
        fitness = []
        titi = 0
        QL = qlearning()
        flag = 1
        for t in range(self.max_iter):
            titi += 1
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.X[i])
                #print('temp: ',temp)
                #print('x: ',self.X[i])
                if temp < self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i].copy()
                    if self.p_fit[i] < self.fit:  # 更新全局最优
                        self.gbest = self.X[i].copy()
                        self.fit = self.p_fit[i]
            if titi == self.checkperiod:
                titi = 0
                state = self.statecheck()
                action = QL.get_action(state)
                self.action_do(action)
                flag = 0
            if titi == self.checkperiod-1 and flag == 0:
                reward = self.rewardcheck()
                next_state = self.statecheck()
                self.statememory()
                QL.Qfresh(state,action,next_state,reward)
            for i in range(self.pN):
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
            print(self.gbest, end=" ")
            print(self.fit, end=' ')  # 输出最优值
            print('time', t)
            #print(QL.Q)
        return fitness
 
        # ----------------------程序执行-----------------------
 
 
my_pso = PSO(pN=10, dim=3, max_iter=800)
my_pso.init_Population()
fitness = my_pso.iterator()
# -------------------画图--------------------
plt.figure(1)
plt.title("Q-pso")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, 800)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()