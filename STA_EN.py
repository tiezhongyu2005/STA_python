# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:57:13 2021

@author: Intelligent Control and Optimization Decision Group, School of Automation, Central South University

Email: michael.x.zhou@csu.edu.cn
"""

import numpy as np
import time


class STA():
    def __init__(self, funfcn, Range, Maxiter, SE=30, dict_opt={'alpha': 1, 'beta': 1, 'gamma': 1, 'delta': 1}):
        """
        Initialize parameters and initial solution for the state transition algorithm.

        Parameters:
            funfcn: Objective function to be optimized (minimized).
            Range: Search range, 2D array with first row as lower bounds and second row as upper bounds.
            Maxiter: Maximum number of iterations, controlling the algorithm's run duration.
            SE: Sample size, defaults to 30, determining the number of candidate solutions per iteration.
            dict_opt: Dictionary of algorithm hyperparameters, including alpha (rotation step size), beta (translation factor),
                      gamma (expansion strength), and delta (axial perturbation magnitude), defaults to 1 for all.
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
        """Initialize the best solution and history record."""
        self.fBest = self.funfcn(self.Best)
        self.history = []
        self.history.append(self.fBest)

    def op_rotate(self):
        """Generate new candidate solutions via rotation operation."""
        R1 = np.random.uniform(-1, 1, (self.SE, self.Dim))
        R2 = np.random.uniform(-1, 1, (self.SE, 1))
        a = np.tile(self.Best, (self.SE, 1))
        b = np.tile(R2, (1, self.Dim))
        c = R1 / np.tile(np.linalg.norm(R1, axis=1, keepdims=True), (1, self.Dim))
        State = a + self.dict_opt['alpha'] * b * c
        return State

    def op_expand(self):
        """Generate new candidate solutions via expansion operation."""
        a = np.tile(self.Best, (self.SE, 1))
        b = np.random.randn(self.SE, self.Dim)
        State = a + self.dict_opt['gamma'] * b * a
        return State

    def op_axes(self):
        """Generate new candidate solutions via axial perturbation."""
        State = np.tile(self.Best, (self.SE, 1))
        index = np.random.randint(0, self.Dim, size=self.SE)
        A = np.zeros((self.SE, self.Dim))
        A[np.arange(self.SE), index] = 1
        R = np.random.randn(self.SE, self.Dim)
        State = State + self.dict_opt['delta'] * R * A * State
        return State

    def op_translate(self, oldBest):
        """Generate new candidate solutions via translation operation."""
        a = np.tile(self.Best, (self.SE, 1))
        b = np.random.uniform(0, 1, (self.SE, 1))
        c = self.dict_opt['beta'] * (self.Best - oldBest) / (np.linalg.norm(self.Best - oldBest))
        State = a + b * c
        return State


    def fitness(self, State):
        """Evaluate population fitness."""
        fState = np.zeros(self.SE)
        for i in range(self.SE):
            fState[i] = self.funfcn(State[i, :])
        return fState

    def selection(self, State, fState):
        """Returns the optimal solution and its objective value"""
        index = np.argmin(fState)
        return State[index, :], fState[index]

    def bound(self, State):
        """Constrain candidate solutions within the search range."""
        Pop_Lb = np.tile(self.Range[0], (State.shape[0], 1))
        Pop_Ub = np.tile(self.Range[1], (State.shape[0], 1))
        repBest = np.tile(self.Best, (State.shape[0], 1))
        idxLow = State < Pop_Lb
        idxHigh = State > Pop_Ub
        State[idxLow] = 0.5 * (repBest[idxLow] + Pop_Lb[idxLow])
        State[idxHigh] = 0.5 * (repBest[idxHigh] + Pop_Ub[idxHigh])
        return State

    def updateBest(self, operation):
        """Execute the specified operation and update the best solution"""
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
        """Run the state transition algorithm, iterating and recording results."""
        time_start = time.time()
        self.initialization()
        for i in range(self.Maxiter):
            self.updateBest(self.op_rotate)
            self.updateBest(self.op_expand)
            self.updateBest(self.op_axes)
            self.history.append(self.fBest)
            print("Iteration {} best value: {:.5e}".format(i, self.fBest))
            if (abs(self.history[i + 1] - self.history[i]) < 1e-20) & (self.dict_opt['alpha'] < 1e-8):
                break

            if self.dict_opt['alpha'] < 1e-8:
                self.dict_opt['alpha'] = self.alpha
            else:
                self.dict_opt['alpha'] = 0.5 * self.dict_opt['alpha']
   
        self.history = np.array(self.history)
        time_end = time.time()
        print("Total running time: {:.5f} s".format(time_end - time_start))
