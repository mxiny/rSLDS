import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def load_data(path):
    data = np.load(path, allow_pickle=True)
    z = data['Z']
    x = data['X']
    y = data['Y']
    return z, x, y

def normalize(x):
    return (x - x.mean(axis=0)) / (np.linalg.norm(x) + 1e-8)

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def lower_pass(x):
    b, a = signal.butter(8, 0.05, 'lowpass')
    x_filted = signal.filtfilt(b, a, x, axis=0)
    return x_filted

def plot_comparison(y, y_est):
    T, N = y.shape
    f, axs = plt.subplots(N, 1)
    # Each dimension matching
    for i in range(N):
        axs[i].plot(range(T), y[:, i], color='k')
        axs[i].plot(range(T), y_est[:, i], color='r')
        axs[i].set_title('%d dimension' %i)
        if i != N - 1:
            axs[i].set_xticks([])

    plt.show()


def plot_prediction_comparison(y_true, y_est, pred_len):
    # Normalize
    y_true_norm = normalize(y_true)
    y_est_norm = normalize(y_est)

    # Plot each dimension
    T, N = y_true.shape
    f, axs = plt.subplots(N, 1)
    for i in range(4):
        y_est_norm[:, i] = lower_pass(y_est_norm[:, i]) 
        axs[i].plot(range(T), y_true_norm[:, i], color='k')
        axs[i].plot(range(T - pred_len), y_est_norm[:T - pred_len, i], color='b')
        axs[i].plot(range(T - pred_len, T), y_est_norm[-pred_len, i], color='r')
        axs[i].set_title('%d dimension' % i)
        if i != 3:
            axs[i].set_xticks([])
    
    plt.show()
