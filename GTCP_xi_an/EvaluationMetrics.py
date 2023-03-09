from sklearn import metrics
# from cvxopt import spmatrix, sqrt, base
import torch
import numpy as np
import math


class Evaluation_Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def accuracy(target, output):
        """
        :param output: predictions
        :param target: ground truth
        :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
        """
        return 1 - torch.linalg.norm(target - output, 'fro') / torch.linalg.norm(target, 'fro')

    @staticmethod
    def r2(target, output):
        """
        :param output: predictions
        :param target: ground truth
        :return: R square (coefficient of determination)
        """
        return 1 - torch.sum((target - output) ** 2) / torch.sum((target - torch.mean(output)) ** 2)

    @staticmethod
    def MSE(target, output):
        return metrics.mean_squared_error(target, output)

    @staticmethod
    def MAE(target, output):
        # return np.mean(np.abs(target - output))
        return metrics.mean_absolute_error(target, output)

    @staticmethod
    def MAPE(target, output):
        # return np.mean(np.abs(target - output) / (target + 5))
        diff = np.abs(np.array(target) - np.array(output))
        return np.mean(diff / target)

    @staticmethod
    def SMAPE(target, output):
        return 2.0 * np.mean(np.abs(output - target) / (np.abs(output) + np.abs(target))) * 100

    @staticmethod
    def RMSE(target, output):
        # return np.sqrt(np.mean(np.power(target - output, 2)))
        return np.sqrt(metrics.mean_squared_error(target, output))

    @staticmethod
    def total(target, output):
        mae = Evaluation_Utils.MAE(target, output)
        # mape = Evaluation_Utils.MAPE(target, output)
        rmse = Evaluation_Utils.RMSE(target, output)
        return mae, rmse
