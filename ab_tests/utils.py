import numpy as np
from scipy.stats import beta
from matplotlib import pyplot as plt


def posterior_samples(imp, conv, n_samples=100000):
    posterior = beta(1 + conv, 1 + imp - conv)
    samples = posterior.rvs(n_samples)
    return posterior, samples


def plot_function_densities(funcs, labels, x=np.linspace(0, 1, 500)):
    for func, label in zip(funcs, labels):
        plt.plot(x, func.pdf(x), label=label)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()


def relative_increase(a, b):
    return (a - b) / b


def plot_sample_densities(arr, labels, bins=50):
    for i, label in enumerate(labels):
        plt.hist(arr[i], label=label, bins=bins)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
