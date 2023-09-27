##Figure HAN et HAN Solo et CC
## TODO: fix it
import numpy as np
import matplotlib.pyplot as plt
from hansolo import *

answersEWA = np.load("EWA100.npy")
M = 2997
Nb = 100
paq = 180

level = 0.1

plt.figure(figsize=(5.5, 4))

answersEWA2 = np.load("EWA100_inhib.npy")
M = 2997
Nb = 100
paq = 180

level = 0.1

Y = np.zeros(
    (Nb, M - paq)
)  # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M - paq):
        Y[nb, m] = np.sum(answersEWA2[nb, m : m + paq]) / paq


Y_med = np.zeros(M - paq)
Y_90 = np.zeros(M - paq)
Y_10 = np.zeros(M - paq)
Y_max = np.zeros(M - paq)

for m in range(M - paq):
    X = Y[:, m]
    X = np.sort(X)
    Y_med[m] = X[int(Nb / 2)]
    Y_90[m] = X[int(Nb * (1 - level))]
    Y_10[m] = X[int(Nb * level)]
    Y_max[m] = X[Nb - 1]
# if Y_10[m] > Y_med[m]:
# print(m)

X2 = np.arange(M - paq) + paq
plt.plot(X2, Y_med, label="HAN with EWA", color="g")
plt.fill_between(X2, Y_10, Y_90, color="g", alpha=0.2)


answersPWA2 = np.load("PWA100_inhib.npy")
# answersPWA=np.load('PWA100_para_opti.npy')

Y = np.zeros(
    (Nb, M - paq)
)  # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M - paq):
        Y[nb, m] = np.sum(answersPWA2[nb, m : m + paq]) / paq


Y_med = np.zeros(M - paq)
Y_90 = np.zeros(M - paq)
Y_10 = np.zeros(M - paq)


for m in range(M - paq):
    X = Y[:, m]
    X = np.sort(X)
    Y_med[m] = X[int(Nb / 2)]
    Y_90[m] = X[int(Nb * (1 - level))]
    Y_10[m] = X[int(Nb * level)]
    Y_max[m] = X[Nb - 1]
# if Y_10[m] > Y_med[m]:
# print(m)

X2 = np.arange(M - paq) + paq

plt.plot(X2, Y_med, label="HAN with PWA", color="magenta")

plt.fill_between(X2, Y_10, Y_90, color="magenta", alpha=0.1)


Y = np.zeros(
    (Nb, M - paq)
)  # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M - paq):
        Y[nb, m] = np.sum(answersEWA[nb, m : m + paq]) / paq


Y_med = np.zeros(M - paq)
Y_90 = np.zeros(M - paq)
Y_10 = np.zeros(M - paq)
Y_max = np.zeros(M - paq)

for m in range(M - paq):
    X = Y[:, m]
    X = np.sort(X)
    Y_med[m] = X[int(Nb / 2)]
    Y_90[m] = X[int(Nb * (1 - level))]
    Y_10[m] = X[int(Nb * level)]
    Y_max[m] = X[Nb - 1]
# if Y_10[m] > Y_med[m]:
# print(m)

X2 = np.arange(M - paq) + paq
plt.plot(X2, Y_med, label="HAN Solo with EWA", color="b")
plt.fill_between(X2, Y_10, Y_90, color="b", alpha=0.1)


answersPWA = np.load("PWA100.npy")
# answersPWA=np.load('PWA100_para_opti.npy')

Y = np.zeros(
    (Nb, M - paq)
)  # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M - paq):
        Y[nb, m] = np.sum(answersPWA[nb, m : m + paq]) / paq


Y_med = np.zeros(M - paq)
Y_90 = np.zeros(M - paq)
Y_10 = np.zeros(M - paq)


for m in range(M - paq):
    X = Y[:, m]
    X = np.sort(X)
    Y_med[m] = X[int(Nb / 2)]
    Y_90[m] = X[int(Nb * (1 - level))]
    Y_10[m] = X[int(Nb * level)]
    Y_max[m] = X[Nb - 1]
# if Y_10[m] > Y_med[m]:
# print(m)

X2 = np.arange(M - paq) + paq

plt.plot(X2, Y_med, label="HAN Solo with PWA", color="orange")

plt.fill_between(X2, Y_10, Y_90, color="orange", alpha=0.2)

answersCC = np.load("CC.npy")
# answersPWA=np.load('PWA100_para_opti.npy')

Y = np.zeros(
    (Nb, M - paq)
)  # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M - paq):
        Y[nb, m] = np.sum(answersCC[nb, m : m + paq]) / paq


Y_med = np.zeros(M - paq)
Y_90 = np.zeros(M - paq)
Y_10 = np.zeros(M - paq)


for m in range(M - paq):
    X = Y[:, m]
    X = np.sort(X)
    Y_med[m] = X[int(Nb / 2)]
    Y_90[m] = X[int(Nb * (1 - level))]
    Y_10[m] = X[int(Nb * level)]
    Y_max[m] = X[Nb - 1]
# if Y_10[m] > Y_med[m]:
# print(m)

X2 = np.arange(M - paq) + paq

plt.plot(X2, Y_med, label="Component Cue", color="brown")

plt.fill_between(X2, Y_10, Y_90, color="brown", alpha=0.2)


plt.yticks([0.6, 0.8, 1])
plt.xticks([500, 1500, 3000])
plt.ylabel("Proportion of correction classifications")
plt.xlabel("m")

plt.legend()
plt.savefig("curves.pdf")
plt.show()
