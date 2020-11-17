import numpy as np
import random
import matplotlib.pyplot as plt
import math
 
 
# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, pN, dim, max_iter):
        self.w = 0.2
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
        self.L = 0.03

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
        for i in range(self.dim-5):
            ta += self.tri[i][0] * self.X[s][i]
            tb += self.tri[i][1] * self.X[s][i]
            tc += self.tri[i][2] * self.X[s][i]
        for i in range(7,10):
            na += self.nor[i-7][0] * self.X[s][i]
            nb += self.nor[i-7][1] * self.X[s][i]
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
        for i in range(2500):
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
        p = self.u(r)
        up = p ** 0.61
        ua = (1-p) ** 0.61
        bottom = (up + ua) ** (1 / 0.61)
        return up / bottom
    
    def wsq(self, r):
        p = self.u(r)
        up = p ** 0.65
        ua = (1-p) ** 0.65
        bottom = (up + ua) ** (1 / 0.65)
        return up / bottom
    
    def ws(self, r):
        p = self.u(r)
        up = p ** 0.69
        ua = (1-p) ** 0.69
        bottom = (up +ua) ** (1 / 0.69)
        return up / bottom

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
        for i in range(self.dim - 2):
            if self.X[s][i] <= 0:
                #print("before: ",self.X[s][i])
                self.X[s][i] = 0.0001
                #print("now: ",self.X[s][i])
                #print(" --changing...")
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
            #print('they are: ',self.X[i])
            tmp = self.vall(0)
            self.p_fit[i] = tmp
            if tmp > self.fit:
                self.fit = tmp
                self.gbest = self.X[i].copy()
            
                # ----------------------更新粒子位置----------------------------------
 
    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            for i in range(self.pN):  # 更新gbest\pbest
                #print('pn:', i,'they:', self.X[i])
                temp = self.vall(i)
                #print('now,vall: ', temp)
                if temp > self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i].copy()
                    if self.p_fit[i] > self.fit:  # 更新全局最优
                        #print('gbest changed')
                        self.gbest = self.X[i].copy()
                        self.fit = self.p_fit[i]
            for i in range(self.pN):
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
                self.popfresh(i)
                self.mix(i)
                self.popfresh2(i)
                s = self.dispersion(i)
                while s:
                    self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                                self.c2 * self.r2 * (self.gbest - self.X[i])
                    self.X[i] = self.X[i] + self.V[i]
                    self.popfresh(i)
                    self.mix(i)
                    self.popfresh2(i)
                    s = self.dispersion(i)
            #for i in range(self.pN):
              #  print('pn:', i,'they:', self.X[i])
            fitness.append(self.fit)
            print(self.gbest, end=" ")
            print(self.fit, end=' ')  # 输出最优值
            print('time:', t)
        return fitness
 
        # ----------------------程序执行-----------------------
 
 
my_pso = PSO(pN=20, dim=12, max_iter=100)
my_pso.init_Population()
fitness = my_pso.iterator()

plt.figure(1)
plt.title("original PSO PT-FPS")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, 100)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()
