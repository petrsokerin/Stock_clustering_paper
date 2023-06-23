from scipy.optimize import minimize
import numpy as np
import pandas as pd
from datetime import datetime

class MarkowitzPortfolio:
    def __init__(self, returns, cov_mat, ret_det=None, kargs=None):
        self.returns = returns
        self.cov_mat = cov_mat
        if ret_det is not None:
            self.ret_det = ret_det
        elif kargs is not None and 'ret_det' in kargs.keys():
            self.ret_det = kargs['ret_det']
        else:
            raise ValueError('ret_det or kargs must be set')


    def fit(self):

        def objective(x):  # функция риска
            return np.array(x).T @ self.cov_mat @ np.array(x)

        def constraint1(x):  # условие для суммы долей -1
            return 1.0 - np.sum(np.array(x))

        def constraint2(x):  # задание доходности
            return self.returns.T @ np.array(x) - self.ret_det

        n = len(self.returns)
        x0 = [1/n]*n  # начальное значение переменных для поиска минимума функции риска
        b = (0.0, 0.3)  # условие для  x от нуля до единицы включая пределы
        bnds = [b] * n  # передача условий в функцию  риска(подготовка)
        con1 = {'type': 'eq', 'fun': constraint1}  # передача условий в функцию  риска(подготовка)
        con2 = {'type': 'eq', 'fun': constraint2}  # передача условий в функцию  риска(подготовка)
        cons = [con1, con2]  # передача условий в функцию  риска(подготовка)
        sol = minimize(objective, x0, method='SLSQP', \
                       bounds=bnds, constraints=cons)

        status = sol.message
        weights = sol.x
        weights = [np.round(x, 7) for x in weights]
        weights = weights / np.sum(weights)
        
        return weights, status


