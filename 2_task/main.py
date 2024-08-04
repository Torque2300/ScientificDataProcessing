import numpy as np
import matplotlib.pyplot as plt


def hertz(length, sampling):
    y = [0.0] * length
    td = 1 / sampling
    for i in range(len(y)):
        y[i] = i * td
    z = [0.0] * length
    for i in range(len(y)):
        z[i] = i / (td * len(y))
    return z


def create_func_values(function, n, period, step):
    res = np.zeros(n)
    tmp = 0
    for i in range(n):
        if tmp <= period:
            res[i] = function(tmp)
            tmp += step
        if tmp > period:
            tmp = tmp % period
    return res


def DFT_slow(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def FFT(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if (N <= 1):
        return x
    fourier = np.zeros(N, dtype=complex)
    odd = FFT(x[1::2])
    even = FFT(x[::2])
    for i in range(N // 2):
        fourier[i] = even[i] + np.exp((-2j * np.pi * i) / N) * odd[i]
        fourier[N // 2 + i] = even[i] - np.exp((-2j * np.pi * i) / N) * odd[i]
    return fourier


def main():
    N = 200  # Количество отсчетов
    fmax = 800  # максимальная частота
    T = 1.0 / fmax  #
    f = 100  # частота сигнала в Гц

    x = np.linspace(0.0, N * T, N // 2)
    y = np.cos(f * 2.0 * np.pi * x)
    yf = DFT_slow(y)
    xf = np.linspace(0.0, fmax / 2, N // 2)
    yff = np.fft.fft(y)
    rft_s = np.fft.ifft(yf)
    rft = np.fft.ifft(yff)


    noised_ft = np.fft.fft(y + np.random.normal(0, 1, x.shape))
    rnoised_ft = np.fft.ifft(noised_ft)

    fig = plt.figure(figsize=(15, 15))  # размер полотна
    plt.subplots_adjust(wspace=0.7, hspace=0.7)  # отступ между графиками

    # plt.subplot(351)
    # plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    # plt.grid()
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Амплитуда - DFT-Slow прямое')
    #
    # plt.subplot(352)
    # plt.plot(xf, 2.0 / N * np.abs(yff[0:N // 2]))
    # plt.grid()
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Амплитуда - FFT.FFT')
    #
    # plt.subplot(353)
    # plt.plot(xf, y[0:N // 2])
    # plt.grid()
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Амплитуда - Косинус')
    #
    # plt.subplot(354)
    # plt.plot(xf, rft_s[0:N // 2])
    # plt.grid()
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Амплитуда - Обратное DFT-Slow')
    #
    # plt.subplot(355)
    # plt.plot(xf, rft[0:N // 2])
    # plt.grid()
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Амплитуда - Обратное fft.ifft')
    #
    # plt.subplot(356)
    # plt.plot(xf, noised_ft[0:N // 2])
    # plt.grid()
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Амплитуда - fft with noise')
    #
    # plt.subplot(357)
    # plt.plot(xf, rnoised_ft[0:N // 2])
    # plt.grid()
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Амплитуда - Обратное от прямого с шумом')
    #
    # A = 2
    # T = 4
    # time_interval = hertz(20, 4)
    # meandr = create_func_values(lambda x: A if x < T / 2 else -A, 20, T, 1/4)
    # mft_s = DFT_slow(meandr)
    # mft = np.fft.fft(meandr)
    # mft_noised = np.abs(np.fft.fft(meandr + np.random.normal(0, 1, 20)))
    #
    # plt.subplot(358)
    # plt.plot(time_interval, meandr)
    # plt.grid()
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Амплитуда прямоугольного сигнала ')
    #
    # plt.subplot(359)
    # plt.plot(time_interval, mft_s)
    # plt.grid()
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Амплитуда DFT_SLOW Meander ')
    #
    # plt.subplot(3, 5, 10)
    # plt.plot(time_interval, mft)
    # plt.grid()
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Амплитуда FFT Meander ')
    #
    # plt.subplot(3, 5, 11)
    # plt.plot(time_interval, mft_noised)
    # plt.grid()
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Амплитуда FFT Noised Meander')
    #
    # plt.subplot(3, 5, 12)
    # plt.plot(np.abs(FFT(y)))
    # plt.grid()
    # plt.plot(np.abs(np.fft.fft(y)))
    # plt.plot(np.abs(DFT_slow(y)))

    plt.plot(xf, np.abs(FFT(y))[:N // 2], '+')
    plt.grid()
    plt.xlabel('Частота, Гц')
    plt.ylabel('Амплитуда')
    plt.plot(xf, np.abs(np.fft.fft(y))[: N // 2], '*')
    plt.plot(xf, np.abs(DFT_slow(y))[: N // 2], '-')
    plt.show()


if __name__ == '__main__':
    main()
