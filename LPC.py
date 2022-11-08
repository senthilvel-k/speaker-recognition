# calculate LPC coefficients from sound file
from __future__ import division
import numpy as np


def autocorr(x):
    n = len(x)
    variance = np.var(x)
    x = x - np.mean(x)
    # n numbers from last index
    r = np.correlate(x, x, mode='full')[-n:]
    result = r / (variance * (np.arange(n, 0, -1)))
    return result


def createSymmetricMatrix(acf, p):
    R = np.empty((p, p))
    for i in range(p):
        for j in range(p):
            R[i, j] = acf[np.abs(i - j)]

    return R


def lpc(s, fs, p):
    # divide into segments of 25 ms with overlap of 10ms
    nSamples = np.int32(0.025 * fs)
    overlap = np.int32(0.01 * fs)
    nFrames = np.int32(np.ceil(len(s) / (nSamples - overlap)))

    # zero padding to make signal length long enough to have nFrames
    padding = ((nSamples - overlap) * nFrames) - len(s)
    if padding > 0:
        signal = np.append(s, np.zeros(padding))
    else:
        signal = s
    segment = np.empty((nSamples, nFrames))
    start = 0
    for i in range(nFrames):
        segment[:, i] = signal[start:start + nSamples]
        start = (nSamples - overlap) * i

    # calculate LPC with Yule-Walker
    lpc_coeffs = np.empty((p, nFrames))
    for i in range(nFrames):
        acf = autocorr(segment[:, i])
        r = -acf[1:p + 1].T
        R = createSymmetricMatrix(acf, p)
        lpc_coeffs[:, i] = np.dot(np.linalg.inv(R), r)
        lpc_coeffs[:, i] = lpc_coeffs[:, i] / np.max(np.abs(lpc_coeffs[:, i]))

    return lpc_coeffs
