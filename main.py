import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np


def fourier_transform(lst):
    k = 0
    n = 0
    four_series = [0] * (len(lst))
    while k < len(lst):
        res = 0
        while n < len(lst):
            res += lst[n]*np.exp(2*np.pi*(-1)*k*n*1j/len(lst))
            n += 1
        four_series[k] = round(res.real, 2)+round(res.imag, 2) * 1j
        n = 0
        k += 1
    return four_series


def create_func_values(function, n, begin, end, step):
    res = np.zeros(n)
    for i in range(n):
        if begin <= end:
            begin += step
            res[i] = function(begin)
    return res


T = 2 * np.pi


def coefs(n, f, coef_function):
    coefs = np.zeros(n)
    for i in range(n):
        coefs[i] = coef_function(i, f)
    return coefs


def a_n(n, f):
    return integrate.quad(lambda x: 2 * f(x) * np.cos(n * x * 2 * np.pi / T) / T, 0, T)[0]


def b_n(n, f):
    return integrate.quad(lambda x: 2 * f(x) * np.sin(n * x * 2 * np.pi / T) / T, 0, T)[0]


def fourier_series(t, ai, bi):
    fourier = np.zeros(len(t))
    for i in range(len(t)):
        fourier[i] = ai[0] / 2
        for j in range(1, len(ai)):
            fourier[i] += ai[j] * np.cos(j * 2 * np.pi / T * t[i]) + bi[j] * np.sin(j * 2 * np.pi / T * t[i])
    return fourier


def main():
    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(15)

    a_cos = coefs(10, np.cos, a_n)
    b_cos = coefs(10, np.cos, b_n)

    plt.subplot(2, 5, 1)
    cos = create_func_values(np.cos, 100, 0, 2 * np.pi, 2 * np.pi / 100)
    fourier = fourier_series(np.arange(0, 2 * np.pi, 2 * np.pi / 100), a_cos, b_cos)
    plt.xlabel('t')
    plt.ylabel('f(t) - cos')
    plt.plot(cos, label='function')
    plt.plot(fourier, '.', label='fourier')
    plt.legend()

    a = coefs(10, lambda x: 1 if x < T / 2 else -1, a_n)
    b = coefs(10, lambda x: 1 if x < T / 2 else -1, b_n)

    plt.subplot(2, 5, 2)
    mean = create_func_values(lambda x: 1 if x < T / 2 else -1, 100, 0, 2 * np.pi, 2 * np.pi / 100)
    fourier = fourier_series(np.arange(0, 2 * np.pi, 2 * np.pi / 100), a, b)
    plt.xlabel('t')
    plt.ylabel('f(t) - rect')
    plt.plot(fourier, '.', label='fourier')
    plt.plot(mean, label='func')
    plt.legend()

    plt.subplot(2, 5, 3)
    plt.xlabel('t')
    plt.ylabel('dt(t)')
    plt.plot(np.abs(mean - fourier))

    y = [0.0] * 100
    td = 1 / 30
    fd = 1 / td
    w = 1
    for i in range(len(y)):
        y[i] = i * td

    four_list = [0.0] * 100

    for i in range(100):
        four_list[i] = np.cos(2 * np.pi * w * i * td)

    four_list = fourier_transform(four_list)

    cutted_four_list = [0.0] * 100
    for i in range(50):
        cutted_four_list[i] = four_list[i]

    z = [0.0] * 100
    for i in range(len(y)):
        z[i] = i / (td * len(y))

    plt.subplot(2, 5, 4)
    plt.xlabel('t')
    plt.ylabel('four_tran(t)')
    plt.plot(z, cutted_four_list)

    plt.subplot(2, 5, 5)
    plt.xlabel('t')
    plt.ylabel('f(t) - fft with noise')

    four_list_noised = [0.0] * 100
    noise = np.random.sample(100)
    for i in range(100):
        four_list_noised[i] = np.cos(2 * np.pi * w * i * td)

    four_list_noised = fourier_transform(four_list_noised + noise)  # - np.mean(noise)

    cutted_four_list_noised = [0.0] * 100
    for i in range(50):
        cutted_four_list_noised[i] = four_list_noised[i]

    plt.plot(z, cutted_four_list_noised)

    a = coefs(100, lambda x: 1 if x < T / 2 else -1, a_n)
    b = coefs(100, lambda x: 1 if x < T / 2 else -1, b_n)
    plt.subplot(2, 5, 6)
    plt.xlabel('t')
    plt.ylabel('f(t) - fft with noise')
    plt.plot(z,a)


    plt.show()
#     видимо 30  Kotelnikov's frequency
# появился шум, увеличился пик раза в два, чтобы хоть как-то исправить это можно отнять среднее (np.mean(noise)), тогда амплитуда
# станет прежней



main()

