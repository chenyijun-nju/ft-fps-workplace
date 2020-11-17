import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
from tensorflow.keras import models, layers, optimizers

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
 
# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, pN, dim, max_iter):
        self.trinum = 7
        self.gusnum = 3
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = random.random()
        self.r2 = random.random()
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 0.0  # 全局最佳适应值
        self.fuzpoint = 50
        self.dbfpt = 2500
        self.EXP = [0.078,0.231,0.119,0.346,0.061,0.122,-0.006,0.199,0.250,0.032]
        self.tri = [[-0.129,0.091,0.259],[-0.280,0.254,0.689],[-0.306,0.161,0.461],[-0.462,0.487,0.872],[-0.071,0.083,0.148],[-0.109,0.164,0.267],[-0.263,-0.077,0.085]]
        self.nor = [[0.199,0.099],[0.250,0.124],[0.032,0.015]]
        self.Mpoint = np.zeros(self.dbfpt)
        self.Mdegree = np.zeros(self.dbfpt)
        self.thou = 0.05
        self.L = 0.02
        self.pbeststate = np.zeros((self.pN, self.dim))# 存储上一周期的pbest
        self.gbeststate = np.zeros((1, self.dim))
        self.checkperiod = 2# 收敛状态检查周期
        self.Xbefore = np.zeros((self.pN, self.dim))# 存储DQN上一次的坐标
        self.pbestbefore = np.zeros((self.pN, self.dim))
        self.gbestbefore = np.zeros((1, self.dim))
        self.distence = 1
        self.dqn_v = 0

    # ---------------------目标函数-----------------------------
    
    def triangle(self, sup, step, a, b, c):
        y = sup + step
        if y < a:
            ans = 0
        elif y > a and y <= b:
            ans = ((y - a) / (b - a))
        elif y > b and y <= c:
            ans = ((c - sup) / (c - b))
        else:
            ans = 0
        return ans

    def normal(self, sup, step, na, nb):
        y = sup + step
        if y < na:
            ans = math.exp(-(y-na) * (y-na) / nb * nb)
        elif y > na and sup < na:
            ans = 1.0
        elif sup > na:
            ans = math.exp(-(sup - na) * (sup - na) / nb * nb)
        return ans

    def mix(self, s):
        ta = 0
        tb = 0
        tc = 0
        na = 0
        nb = 0
        for i in range(self.trinum):
            ta += self.tri[i][0] * self.X[s][i]
            tb += self.tri[i][1] * self.X[s][i]
            tc += self.tri[i][2] * self.X[s][i]
        for i in range(self.trinum, self.trinum +self.gusnum):
            na += self.nor[i-self.trinum][0] * self.X[s][i]
            nb += self.nor[i-self.trinum][1] * self.X[s][i]
        for i in range(self.fuzpoint):
            sup0 = random.uniform(ta + i * (tc - ta) / self.fuzpoint, ta + (i + 1) * (tc - ta) / self.fuzpoint)
            membership1 = self.triangle(sup0, (tc - ta) / self.fuzpoint, ta, tb, tc)
            for j in range(self.fuzpoint):
                sup1 = random.uniform((na - 3 * nb) + j * nb * 0.12, (na - 3 * nb) + (j + 1) * nb * 0.12)
                membership2 = self.normal(sup1, nb * 0.12, na, nb)
                if membership1 < membership2:
                    temp = membership1
                else:
                    temp = membership2
                self.Mpoint[i * self.fuzpoint + j] = sup0 + sup1
                self.Mdegree[i * self.fuzpoint + j] = temp

    def u(self,r):
        sup1 = 0
        sup2 = sup1
        for i in range(self.dbfpt):
            if self.Mpoint[i] > r and self.Mdegree[i] > sup1:
                sup1 = self.Mdegree[i]
            elif self.Mpoint[i] <= r and self.Mdegree[i] >sup2:
                sup2 = self.Mdegree[i]
        if sup1 >= sup2:
            membership = sup2
        else:
            membership = sup1
        #print('membership now:',membership)
        return membership

    def crm(self, r):
        sup1 = 0.001
        sup2 = sup1
        for i in range(self.dbfpt):
            if self.Mpoint[i] > r and self.Mdegree[i] > sup1:
                sup1 = self.Mdegree[i]
            elif self.Mpoint[i] <= r and self.Mdegree[i] > sup2:
                sup2 = self.Mdegree[i]
        if sup1 > sup2:
            sup1 = 1
        else:
            sup2 = 1
        cr = 0.5 * (sup1 + 1 - sup2)
        return cr
    
    def me(self, s):
        mr = self.X[s][self.dim - 2]
        g = self.X[s][self.dim - 1]
        w =0.001
        igt = 0
        y = mr + w 
        while y <= g:
            igt += self.crm(y) * w
            y += w
        e = mr + igt * (g - mr)
        #word = 'me now'
       # print(word, e)
        return e

    def se(self, a):
        #if a < min(self.Mpoint) or a > max(self.Mpoint):
        #    print('error r:',a,'min:',min(self.Mpoint),'max:',max(self.Mpoint))
        if a <= 0.01:
            return 0  
        ans = - a * math.log(a) - (1 - a) * math.log(1-a)
        return ans 
    
    def sen(self, s):
        mr = self.X[s][self.dim - 2]
        g = self.X[s][self.dim - 1]
        #print('mr:', mr,'g: ', g)
        w =0.001
        igt = 0
        y = mr + w 
        while y <= g:
            a = 0.5 * self.u(y)
            igt += self.se(a) * w
            y += w
        return igt
    
    def b(self, r):
        top = self.u(r) * 0.001 *100
        bottom = max(self.Mpoint)-min(self.Mpoint)
        return top/bottom
    
    def wf(self, r):
        yeta = 0.7060
        gamma = 0.4933
        alpha = 0.90
        p = self.u(r)
        s = 2 * p - 1
        up = yeta * (s+1) ** gamma
        ua = (1-s) ** gamma
        bottom = (up + ua) ** (1 / 0.9)
        return up **(1 / 0.9) / bottom
    
    def wsq(self, r):
        yeta = 0.7060
        gamma = 0.4933
        alpha = 0.90
        p = self.u(r)
        s = 2 * p - 1
        up = yeta * (s+1) ** gamma
        ua = (1-s) ** gamma
        bottom = (up + ua) ** (1 / 0.9)
        return up **(1 / 0.9) / bottom
    
    def ws(self, r):
        yeta = 0.7060
        gamma = 0.4933
        alpha = 0.90
        p = self.u(r)
        s = 2 * p - 1
        up = yeta * (s+1) ** gamma
        ua = (1-s) ** gamma
        bottom = (up + ua) ** (1 / 0.9)
        return up **(1 / 0.9) / bottom

    def vf(self,s):
        a = min(self.Mpoint)
        mr = self.X[s][self.dim - 2]
        w =0.001
        igt = 0
        y = a 
        while y <=mr:
            igt += -2.25 * pow(mr-y, 0.88) * self.wf(y) * w
            y += w
        return igt

    def vsq(self, s):
        mr = self.X[s][self.dim - 2]
        g = self.X[s][self.dim - 1]
        w =0.001
        igt = 0
        y = mr + w 
        while y <= g:
            igt += pow(y - mr, 0.88) * self.wsq(y) * w
            y += w
        return igt
    
    def vs(self, s):
        g = self.X[s][self.dim - 1]
        top = max(self.Mpoint)
        w =0.001
        igt = 0
        y = g + w 
        while y <= top:
            igt += pow(y - g, 0.88) * self.ws(y) * w + 2.2 * pow(y - g, 0.88) * self.ws(y) * w
            y += w
        return igt

    
    def vall(self, s):
        f = self.vf(s)
        sq = self.vsq(s)
        ss = self.vs(s)
        suma = f + sq + ss
        #print('vall now: ',suma,'vs now: ',ss,'vsq now: ',sq,'vf now: ',f)
        return suma
    
    def vars(self, s):
        v = 0
        for i in range(self.dim-2):
            v += (self.X[s][i] - 0.1) ** 2
        return v / 10

    def dispersion(self, s):
        e = self.me(s)
        ea = self.sen(s)
        La = e - self.thou * ea
        v = self.vars(s)
        #print('me: ', e,'ea: ', ea,'La: ', La,'vars: ',v)
        if La < self.L or v > 0.06:
            return True
        else:
            #print('OK')
            return False
    
    def popfresh(self, s):
        #print('popfreshing...')
        e = 0
        sumi = 0
        maxi = 0
        if self.vars(s) >=0.06:
            for i in range(self.dim - 2):
                if self.X[s][i] >= maxi:
                    index = i
            self.X[s][index] = 0.95 * self.X[s][index]
        for i in range(self.dim - 2):
            if self.X[s][i] <= 0:
                self.X[s][i] = 0.0001
            sumi += self.X[s][i]
        for i in range(self.dim - 2):
            self.X[s][i] = self.X[s][i] / sumi
        #print('popfresh solution: ',self.X[s])
    
    def popfresh2(self,s):
        e = 0
        sumi = 0
        b = min(self.Mpoint)
        a = max(self.Mpoint)
        ship = 0
        if b >= 0:
            ship = b
        for i in range(self.dim - 2):
            e += self.X[s][i] * self.EXP[i]
        if self.X[s][self.dim - 2] <= 0 or self.X[s][self.dim - 2] > e or self.X[s][self.dim - 2] < b:
            self.X[s][self.dim - 2] = random.uniform(ship, e)
        if self.X[s][self.dim - 1] > a or self.X[s][self.dim - 1] < e:
            self.X[s][self.dim - 1] = random.uniform(e, a)

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
                        self.X[i][j] += single * random.uniform(0, 0.5 * abs(d))
            #print('before moving gbest:', self.gbest)
        elif action == 1:
            for i in range(self.pN):
                for j in range(self.dim):
                    if self.X[i][j] != self.pbest[i][j]:
                        d = self.X[i][j] - self.pbest[i][j]
                        single = d/abs(d)
                        self.X[i][j] += -single * random.uniform(0, 0.01 * abs(d))
    
    def rewardcheck(self):
        if (self.gbest - self.gbeststate).any():
            return 10
        elif (self.pbeststate - self.pbest).any():
            return 2
        elif (self.pbeststate - self.pbest).any() == False:
            return -2
        elif (self.gbest - self.gbeststate).any() == False:
            return -5
        else:
            return 0
        
    def dqnaction_do(self, action):
        if action == 0:
            return 0.001
        elif action == 1:
            return -0.001
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
            e = 0
            suma = 0
            for j in range(self.dim-2):
                self.X[i][j] = random.random()
                suma += self.X[i][j]
                self.V[i][j] = random.uniform(0, 1)
            for j in range(self.dim-2):
                self.X[i][j] = self.X[i][j] / suma
                e += self.EXP[j] * self.X[i][j]
            #print('pop:',i,'e: ',e)
            self.X[i][self.dim-2] = random.uniform(0, e)
            self.mix(i)
            top = max(self.Mpoint)
            a = random.uniform(e, top)
            self.X[i][self.dim-1] = a
            while self.dispersion(i):
                e = 0
                suma = 0
                for j in range(self.dim-2):
                    self.X[i][j] = random.random()
                    suma += self.X[i][j]
                    self.V[i][j] = random.uniform(0, 1)
                for j in range(self.dim-2):
                    self.X[i][j] = self.X[i][j] / suma
                    e += self.EXP[j] * self.X[i][j]
                #print('pop:',i,'e: ',e)
                self.X[i][self.dim-2] = random.uniform(0, e)
                self.mix(i)
                top = max(self.Mpoint)
                a = random.uniform(e, top)
                self.X[i][self.dim-1] = a
            self.pbest[i] = self.X[i].copy()
            print('they are: ',self.X[i])
            tmp = self.vall(0)
            self.p_fit[i] = tmp
            if tmp > self.fit:
                self.fit = tmp
                self.gbest = self.X[i].copy()
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
                temp = self.vall(i)
                if temp > self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i].copy()
                    if self.p_fit[i] > self.fit:  # 更新全局最优
                        print('gbest changed')
                        self.gbest = self.X[i].copy()
                        self.fit = self.p_fit[i]
            if titi == self.checkperiod:
                titi = 0
                state = self.statecheck()
                action_q = QL.get_action(state)
                self.action_do(action_q)
                flag = 0
            if titi == self.checkperiod-1 and flag == 0:
                reward = self.rewardcheck()
                next_state = self.statecheck()
                self.statememory()
                QL.Qfresh(state,action_q,next_state,reward)
            for i in range(self.pN):
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i]) + self.dqn_v
                self.X[i] = self.X[i] + self.V[i]
                if self.w > 0.1: 
                    self.w = 0.8 * self.w
                self.popfresh(i)
                self.mix(i)
                self.popfresh2(i)
                s = self.dispersion(i)
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
                while s:
                    self.V[i] = 0.2 * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                                self.c2 * self.r2 * (self.gbest - self.X[i]) + self.dqn_v
                    self.X[i] = self.X[i] + self.V[i]
                    self.popfresh(i)
                    self.mix(i)
                    self.popfresh2(i)
                    s = self.dispersion(i)
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
            for i in range(self.pN):
                print('pn:', i,'they:', self.X[i])
            fitness.append(self.fit)
            print(self.gbest, end=" ")
            print(self.fit, end=' ')  # 输出最优值
            print('time:', t)
        return fitness
 
        # ----------------------程序执行-----------------------
 
 
my_pso = PSO(pN=15, dim=12, max_iter=1000)
my_pso.init_Population()
fitness = my_pso.iterator()

plt.figure(1)
plt.title("Q-DQN PSO PT-FPS")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, 1000)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()
