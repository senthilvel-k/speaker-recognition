from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import EUDistance
from mel_coefficients import mfcc

from train import trainingh
from trainz import trainingz


nfiltbank = 22

directoryz = 'zero/test'
directoryh='hello/test_hello'
directory1 = 'zero/speaker'
directory2='hello/speaker'


def minDistance(features, codebooks):
    speaker = 0
    distmin = np.inf
    for k in range(np.shape(codebooks)[0]):
        D = EUDistance(features, codebooks[k, :, :])
        dist = np.sum(np.min(D, axis=1)) / (np.shape(D)[0])
        if dist < distmin:
            distmin = dist
            speaker = k
    return speaker

def h1():
    nSpeaker = 7
    nCorrect_MFCC = 0
    for i in range(nSpeaker):
        fname = '/s' + str(i+1) + '.wav'

        print('Now speaker ', str(i + 1), 'features are being tested')
        (fs, s) = read(directoryh + fname)

        mel_coefs = mfcc(s, fs, nfiltbank,h_z)

        sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)


        print('Speaker(hello)', (i + 1), ' in test matches with speaker ', (sp_mfcc + 1), 'in train for training with MFCC')


        if i == sp_mfcc:
            nCorrect_MFCC += 1
    percentageCorrect_MFCC = (nCorrect_MFCC / nSpeaker) * 100
    print('Accuracy of result for training with MFCC is ', percentageCorrect_MFCC, '%')

def h2():
    nSpeaker = 7
    nCorrect_MFCC = 0
    fname = input("enter wav file: ")
    for i in range(nSpeaker):

        print('Now speaker ', str(i + 1), 'features are being tested')
        (fs, s) = read(directory2 + fname)

        mel_coefs = mfcc(s, fs, nfiltbank,h_z)

        sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)


        if i == sp_mfcc:
            nCorrect_MFCC += 1

    names = {1: "surya", 2: "yuvan", 3: "khamalesh", 4: "lekha", 5: "lalitha", 6:"senthil", 7: "shivaritha" }
    print('Identified Speaker:', names[sp_mfcc + 1])

def z1():
    nSpeaker = 8
    nCorrect_MFCC = 0
    for i in range(nSpeaker):
        fname = '/s' + str(i+1) + '.wav'

        print('Now speaker ', str(i + 1), 'features are being tested')
        (fs, s) = read(directoryz + fname)

        mel_coefs = mfcc(s, fs, nfiltbank,h_z)

        sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)

        print('Speaker(zero)', (i + 1), ' in test matches with speaker ', (sp_mfcc + 1), 'in train for training with MFCC')

        if i == sp_mfcc:
            nCorrect_MFCC += 1
    percentageCorrect_MFCC = (nCorrect_MFCC / nSpeaker) * 100
    print('Accuracy of result for training with MFCC is ', percentageCorrect_MFCC, '%')

def z2():
    nSpeaker = 7
    nCorrect_MFCC = 0
    fname = input("enter wav file: ")
    for i in range(nSpeaker):

        print('Now speaker ', str(i + 1), 'features are being tested')
        (fs, s) = read(directory1 + fname)

        mel_coefs = mfcc(s, fs, nfiltbank,h_z)

        sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)

        if i == sp_mfcc:
            nCorrect_MFCC += 1

    names = {1: "gaurav", 2: "khamalesh", 3: "yuvan", 4: "lekha", 5: "lalitha", 6:"shivaritha", 7: "surya", 8: "Senthil" }

    print('Identified Speaker:', names[sp_mfcc + 1])


cont=False
while cont==False:
    h_z = input("enter hello or zero(h/z)")
    if h_z == "h":
        (codebooks_mfcc) = trainingh(nfiltbank)
    else:
        (codebooks_mfcc) = trainingz(nfiltbank)

    c=False
    while c==False:
        inp = input("enter choice:\n1.show percentage(p)\n2.identify speaker(s)\n")
        if inp=="1":
            if h_z=="h":
                h1()
            else:
                z1()
        elif inp=="2":
            if h_z=="h":
                h2()
            else:
                z2()
        y1=input("do u wanna change word? ")
        if y1 == "y":
            c = True
    y=input("do u want to continue?(y/n): ")
    if y=="n":
        cont=True
        break

