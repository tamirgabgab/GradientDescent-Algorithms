import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

import time


def model_parameters(data, lam=3):
    data = np.array(data)
    X = data[:, 1:]
    Y = data[:, 0]

    x_factor = np.sqrt(np.linalg.norm(X))
    y_factor = np.linalg.norm(Y)
    X = X / x_factor
    Y = Y / y_factor

    XTX = np.transpose(X) @ X
    N, d = X.shape

    w, V = np.linalg.eig(XTX)
    L = max(w) / N + lam
    mu = lam

    theta_opt = np.linalg.inv(XTX + N * lam * np.identity(d)) @ np.transpose(X) @ Y

    return X, Y, L, mu, theta_opt


def Gradient_Descent(params, stepsize=lambda L, mu: 1 / L, mc=1, accelerated='none', theta_0=[0]):
    X, Y, L, mu, theta_opt = params

    max_it = 400
    N, d = X.shape

    if not theta_0.all():
        theta_0 = np.random.uniform(-1e+7, 1e+7, d)

    error_practical = np.zeros(max_it + 1)
    error_theoretical = np.zeros(max_it + 1)

    A = (1 / N) * np.transpose(X) @ X + mu * np.identity(d)
    B = (1 / N) * np.transpose(X) @ Y

    f_opt = (np.linalg.norm(X @ theta_opt - Y) ** 2) / (2 * N) + (np.linalg.norm(theta_opt) ** 2) * mu / 2

    gamma = stepsize(L, mu)

    for j in range(mc):

        theta = theta_0
        init_error = norm(X @ theta - Y) ** 2 / (2 * N) + norm(theta) ** 2 * mu / 2 - f_opt
        error_practical[0] += init_error
        if not j:
            if accelerated == 'none':
                if gamma == 1 / L:
                    error_theoretical[0] = init_error
                elif gamma == 2 / (mu + L):
                    error_theoretical[0] = (L / 2) * norm(theta - theta_opt) ** 2
                else:
                    error_theoretical[0] = 0  # undefined stepsize
            else:
                error_theoretical[0] = ((mu + L) / 2) * norm(theta - theta_opt) ** 2

        for t in range(max_it):
            prev_theta = np.copy(theta)
            gradient = A @ theta - B
            theta -= gamma * gradient

            if accelerated == "Nesterov83":
                beta = 1 - 2 / (np.sqrt(L / mu) + 1)
            elif accelerated == "Heavy-ball":
                beta = (1 - 2 / (np.sqrt(L / mu) + 1)) ** 2
            else:
                beta = 0

            theta += beta * (theta - prev_theta)

            error_practical[t + 1] += norm(X @ theta - Y) ** 2 / (2 * N) + norm(theta) ** 2 * mu / 2 - f_opt
            if not j:
                if accelerated == 'none':
                    if gamma == 1 / L:
                        error_theoretical[t + 1] = np.maximum(0, error_theoretical[t] * (1 - mu / L))
                    elif gamma == 2 / (mu + L):
                        error_theoretical[t + 1] = np.maximum(0, error_theoretical[t] * ((L - mu) / (L + mu)) ** 2)
                    else:
                        error_theoretical[t + 1] = 0  # undefined stepsize
                else:
                    error_theoretical[t + 1] = np.maximum(0, error_theoretical[t] * (1 - np.sqrt(mu / L)))

    return error_practical / mc, error_theoretical


def Coordinate_Descent(params, stepsize=lambda L, mu: 1 / L, mc=1, approach='Cyclic Order', theta_0=[0]):
    X, Y, L, mu, theta_opt = params

    max_it = 400
    N, d = X.shape
    if not theta_0.all():
        theta_0 = np.random.uniform(-1e+7, 1e+7, d)

    error_practical = np.zeros(max_it + 1)
    error_theoretical = np.zeros(max_it + 1)

    XT = np.transpose(X)
    Xi_norms = [norm(XT[i, :]) ** 2 for i in range(d)]
    f_opt = (np.linalg.norm(X @ theta_opt - Y) ** 2) / (2 * N) + (np.linalg.norm(theta_opt) ** 2) * mu / 2

    gamma = stepsize(L, mu)

    for j in range(mc):
        theta = theta_0
        init_error = norm(X @ theta - Y) ** 2 / (2 * N) + norm(theta) ** 2 * mu / 2 - f_opt
        error_practical[0] += init_error
        if not j:
            error_theoretical[0] = init_error

        for t in range(max_it):

            r = Y - X @ theta

            if approach == 'Cyclic Order':
                for i in range(d):
                    theta[i] = (XT[i, :] @ r + theta[i] * Xi_norms[i]) / (Xi_norms[i] + N * mu)

            elif approach == 'Random Sampling':
                i = np.random.randint(d)
                theta[i] = (XT[i, :] @ r + theta[i] * Xi_norms[i]) / (Xi_norms[i] + N * mu)

            error_practical[t + 1] += norm(X @ theta - Y) ** 2 / (2 * N) + norm(theta) ** 2 * mu / 2 - f_opt
            if not j:
                if approach == 'Cyclic Order':
                    error_theoretical[t + 1] = error_theoretical[t] * (1 - mu / (2 * (d + 1) * L))
                elif approach == 'Random Sampling':
                    error_theoretical[t + 1] = error_theoretical[t] * (1 - mu / (d * L))

    return error_practical / mc, error_theoretical


def plot_semilogy(toplots, labels, path, save=1):
    for (plot, text) in zip(toplots, labels):
        plt.semilogy(plot, label=text, marker='D', markevery=100)

    plt.legend()
    plt.xlabel("episodes")
    plt.title(path)
    plt.ylabel(r'${f(x_t)}$-f*')
    plt.grid()
    if save:
        plt.savefig(f'./{path}.png')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv('YearPredictionMSD.txt')
    params = model_parameters(data)
    print("data and parametes were extracted\n")

    beta1 = lambda x, y: 1 / x
    beta2 = lambda x, y: 2 / (x + y)

    X, _, L, mu, _ = params
    _, d = X.shape
    theta_0 = np.random.uniform(-1e+7, 1e+7, d)

    print(f'L is {L} \n\u03BC is {mu}')
    print(f'\u03b21 = 1/L \n\u03b22 = 1/(\u03BC + L)')

    print(f"\nStarting \u03b21 processes:")
    start_time = time.perf_counter()
    error_GD_beta1, bound_GD_beta1 = Gradient_Descent(params, stepsize=beta1, mc=1, theta_0=np.copy(theta_0))
    print(f"done with GD-beta1: took {time.perf_counter() - start_time} seconds")

    start_time = time.perf_counter()
    error_AGD_beta1, bound_AGD_beta1 = Gradient_Descent(params, stepsize=beta1, mc=1, accelerated="Nesterov83",
                                                        theta_0=np.copy(theta_0))
    print(f"done with AGD-beta1: took {time.perf_counter() - start_time} seconds")

    start_time = time.perf_counter()
    error_CD_CO_beta1, bound_CD_CO_beta1 = Coordinate_Descent(params, stepsize=beta1, approach='Cyclic Order', mc=1,
                                                              theta_0=np.copy(theta_0))
    print(f"done with CD_CO-beta1: took {time.perf_counter() - start_time} seconds")

    start_time = time.perf_counter()
    error_CD_RS_beta1, bound_CD_RS_beta1 = Coordinate_Descent(params, stepsize=beta1, approach='Random Sampling', mc=10,
                                                              theta_0=np.copy(theta_0))
    print(f"done with CD_RS-beta1: took {time.perf_counter() - start_time} seconds :: mc = 10")

    print(f"\nStarting \u03b22 processes:")
    start_time = time.perf_counter()
    error_GD_beta2, bound_GD_beta2 = Gradient_Descent(params, stepsize=beta2, mc=1, theta_0=np.copy(theta_0))
    print(f"done with GD-beta2: took {time.perf_counter() - start_time} seconds")

    start_time = time.perf_counter()
    error_AGD_beta2, bound_AGD_beta2 = Gradient_Descent(params, stepsize=beta2, mc=1, accelerated="Heavy-ball",
                                                        theta_0=np.copy(theta_0))
    print(f"done with AGD-beta2: took {time.perf_counter() - start_time} seconds")

    error_CD_CO_beta2, bound_CD_CO_beta2 = Coordinate_Descent(params, stepsize=beta2, approach='Cyclic Order', mc=1,
                                                              theta_0=np.copy(theta_0))
    print(f"done with CD_CO-beta2: took {time.perf_counter() - start_time} seconds")

    start_time = time.perf_counter()
    error_CD_RS_beta2, bound_CD_RS_beta2 = Coordinate_Descent(params, stepsize=beta2, approach='Random Sampling', mc=10,
                                                              theta_0=np.copy(theta_0))
    print(f"done with CD_RS-beta2: took {time.perf_counter() - start_time} seconds :: mc = 10")

    print("\nSection 3:")

    legends = [r"GD - step ${1/L}$", r"AGD - step ${1/L}$", r"CD CO - step ${1/L}$", r"CD RS - step ${1/L}$"]
    plot_semilogy([error_GD_beta1, error_AGD_beta1, error_CD_CO_beta1, error_CD_RS_beta1], legends,
                  "Section3_All_beta1")
    print("figure \'section3_All_beta1\' was saved")

    legends = [r"GD - step ${2/(L+\mu)}$", r"AGD - step ${2/(L+\mu)}$", r"CD CO - step ${2/(L+\mu)}$",
               r"CD RS - step ${2/(L+\mu)}$"]
    plot_semilogy([error_GD_beta2, error_AGD_beta2, error_CD_CO_beta2, error_CD_RS_beta2], legends,
                  "Section3_All_beta2")
    print("figure \'section3_All_beta2\' was saved")

    print("\nSection 4:")
    plot_semilogy([error_GD_beta1, bound_GD_beta1], [r"GD-error-step ${1/L}$", r"GD-bound-step ${1/L}$"],
                  "Section4_GD_beta1")
    plot_semilogy([error_AGD_beta1, bound_AGD_beta1], [r"AGD-error-step ${1/L}$", r"AGD-bound-step ${1/L}$"],
                  "Section4_AGD_beta1")
    plot_semilogy([error_CD_CO_beta1, bound_CD_CO_beta1], [r"CDCO-error-step ${1/L}$", r"CDCO-bound-step ${1/L}$"],
                  "Section4_CDCO_beta1")
    plot_semilogy([error_CD_RS_beta1, bound_CD_RS_beta1], [r"CDRS-error-step ${1/L}$", r"CDRS-bound-step ${1/L}$"],
                  "Section4_CDRS_beta1")
    print("figures \'Section4_GD_beta1\',\'Section4_AGD_beta1\',\'Section4_CDCO_beta1\' & \'Section4_CDRS_beta1\' "
          "were saved")

    plot_semilogy([error_GD_beta2, bound_GD_beta2], [r"GD-error-step ${2/(L+\mu)}$", r"GD-bound-step ${2/(L+\mu)}$"],
                  "Section4_GD_beta2")
    plot_semilogy([error_AGD_beta2, bound_AGD_beta2],
                  [r"AGD-error-step ${2/(L+\mu)}$", r"AGD-bound-step ${2/(L+\mu)}$"], "Section4_AGD_beta2")
    plot_semilogy([error_CD_CO_beta2, bound_CD_CO_beta2],
                  [r"CDCO-error-step ${2/(L+\mu)}$", r"CDCO-bound-step ${2/(L+\mu)}$"], "Section4_CDCO_beta2")
    plot_semilogy([error_CD_RS_beta2, bound_CD_RS_beta2],
                  [r"CDRS-error-step ${2/(L+\mu)}$", r"CDRS-bound-step ${2/(L+\mu)}$"], "Section4_CDRS_beta2")
    print("figures \'Section4_GD_beta2\',\'Section4_AGD_beta2\',\'Section4_CDCO_beta2\' & \'Section4_CDRS_beta2\' "
          "were saved")
