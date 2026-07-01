# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:57:13 2021

@author: 中南大学自动化学院  智能控制与优化决策课题组

Email: michael.x.zhou@csu.edu.cn
"""

import numpy as np
import time


class STA():
    def __init__(self, funfcn, Range, Maxiter, SE=30, dict_opt={'alpha': 1, 'beta': 1, 'gamma': 1, 'delta': 1}):
        """
        初始化状态转移算法的参数和初始解

        参数:
            funfcn: 目标函数，需优化（最小化）的函数
            Range: 搜索范围，二维数组，第一维为下界，第二维为上界
            Maxiter: 最大迭代次数，控制算法运行轮次
            SE: 样本规模，默认为30，决定每次迭代的候选解数量
            dict_opt: 算法超参数字典，包含alpha（旋转步长）、beta（平移因子）、gamma（扩展强度）、delta（轴向扰动幅度），默认值均设为1
        """
        self.funfcn = funfcn
        self.Range = Range
        self.SE = SE
        self.Maxiter = Maxiter
        self.dict_opt = dict_opt.copy()
        self.alpha = self.dict_opt['alpha']
        self.Dim = self.Range.shape[1]
        self.Best = np.array(
            Range[0, :] + ((Range[1, :] - Range[0, :]) * np.random.uniform(0, 1, self.Range.shape[1])))

    def initialization(self):
        """初始化最优解和历史记录"""
        self.fBest = self.funfcn(self.Best)
        self.history = []
        self.history.append(self.fBest)

    def op_rotate(self):
        """通过旋转变换生成新的候选解"""
        R1 = np.random.uniform(-1, 1, (self.SE, self.Dim))
        R2 = np.random.uniform(-1, 1, (self.SE, 1))
        a = np.tile(self.Best, (self.SE, 1))
        b = np.tile(R2, (1, self.Dim))
        c = R1 / np.tile(np.linalg.norm(R1, axis=1, keepdims=True), (1, self.Dim))
        State = a + self.dict_opt['alpha'] * b * c
        return State

    def op_expand(self):
        """通过伸缩变换生成新的候选解"""
        a = np.tile(self.Best, (self.SE, 1))
        b = np.random.randn(self.SE, self.Dim)
        State = a + self.dict_opt['gamma'] * b * a
        return State

    def op_axes(self):
        """通过轴向变换生成新的候选解"""
        State = np.tile(self.Best, (self.SE, 1))
        index = np.random.randint(0, self.Dim, size=self.SE)
        A = np.zeros((self.SE, self.Dim))
        A[np.arange(self.SE), index] = 1
        R = np.random.randn(self.SE, self.Dim)
        State = State + self.dict_opt['delta'] * R * A * State
        return State

    def op_translate(self, oldBest):
        """通过平移变换生成新的候选解"""
        a = np.tile(self.Best, (self.SE, 1))
        b = np.random.uniform(0, 1, (self.SE, 1))
        c = self.dict_opt['beta'] * (self.Best - oldBest) / (np.linalg.norm(self.Best - oldBest))
        State = a + b * c
        return State

    def fitness(self, State):
        """评估种群适应度"""
        fState = np.zeros(self.SE)
        for i in range(self.SE):
            fState[i] = self.funfcn(State[i, :])
        return fState

    def selection(self, State, fState):
        """返回最优解及其目标值"""
        index = np.argmin(fState)
        return State[index, :], fState[index]


    def bound(self, State):
        """将候选解约束在搜索范围内"""
        Pop_Lb = np.tile(self.Range[0], (State.shape[0], 1))
        Pop_Ub = np.tile(self.Range[1], (State.shape[0], 1))
        repBest = np.tile(self.Best, (State.shape[0], 1))
        idxLow = State < Pop_Lb
        idxHigh = State > Pop_Ub
        State[idxLow] = 0.5 * (repBest[idxLow] + Pop_Lb[idxLow])
        State[idxHigh] = 0.5 * (repBest[idxHigh] + Pop_Ub[idxHigh])
        return State

    def updatebest(self, operation):
        """执行operation对应的操作并更新最优解"""
        oldBest = self.Best.copy()
        State = operation()
        State = self.bound(State)
        fState = self.fitness(State)
        newBest, fnewBest = self.selection(State, fState)
        if fnewBest < self.fBest:
            self.fBest, self.Best = fnewBest, newBest
            State = self.bound(self.op_translate(oldBest))
            fState = self.fitness(State)
            newBest, fnewBest = self.selection(State, fState)
            if fnewBest < self.fBest:
                self.fBest, self.Best = fnewBest, newBest

    def run(self):
        """运行状态转移算法，迭代优化并记录结果"""
        time_start = time.time()
        self.initialization()
        for i in range(self.Maxiter):
            self.updatebest(self.op_rotate)
            self.updatebest(self.op_expand)
            self.updatebest(self.op_axes)
            self.history.append(self.fBest)
            print("第{}次迭代的最优值是:{:.5e}".format(i, self.fBest))
            if (abs(self.history[i + 1] - self.history[i]) < 1e-20) & (self.dict_opt['alpha'] < 1e-8):
                break

            if self.dict_opt['alpha'] < 1e-8:
                self.dict_opt['alpha'] = self.alpha
            else:
                self.dict_opt['alpha'] = 0.5 * self.dict_opt['alpha']
 
        self.history = np.array(self.history)
        time_end = time.time()
        print("总耗时:{:.5f}".format(time_end - time_start))


