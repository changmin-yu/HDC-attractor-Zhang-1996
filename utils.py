import numpy as np


def sigmoid(a: float, b: float, c: float, beta: float, x: np.ndarray):
    return a * np.power(np.log(1 + np.exp(b * (x + c))), beta)


def inverse_sigmoid(a: float, b: float, c: float, beta: float, y: np.ndarray):
    return np.log(np.exp(np.power(y / a, 1 / beta)) - 1) / b - c


def directional_tuning(theta: np.ndarray, A: float, B: float, K: float):
    return A + B * np.exp(K * np.cos(theta))