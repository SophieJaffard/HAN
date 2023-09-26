import numpy as np
import scipy.stats as stats
from hansolo import *

Nb = 100
expert = "EWA"  # "EWA" or "PWA" or "Multilin"

M = 2997
dt = 0.002
N = 1000
T = N * dt

freq = 100  # firing rate input neurons +
freq2 = 150

alpha = np.array([100 * dt, 0])  # spontaneous rates of output neurons

p = freq * dt  # spiking proba input neurons
p2 = freq2 * dt

K_input = 12  # nb experts
n_input = 6  # nb input neurons
K_output = 2  # nb output neurons

sup = freq2 * dt * ((9 / 8) + 9)
eta_output = np.sqrt(8 * np.log(K_input) / M) / sup  # para EWA


obj = [
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1],
]
# blue circle blue square blue triangle red circle red square red triangle gray circle gray square gray triangle

obj_tot = []

answers = np.zeros((Nb, M))  # 0 if wrong, 1 if correct

W_output = np.zeros((K_output, K_input, M, Nb))
W_output[:, :, 0, :] += 1 / K_input
F_output = np.zeros((K_output, M, Nb))  # firing rate output neurons

# para_PWA=2*np.log(K_input)
para_PWA = 2


for nb in range(Nb):
    obj2 = objects_random(obj)
    obj_tot.append(obj2)
    L, ind = list_obj(obj2, M)

    W_output_not_renorm = np.zeros((K_output, K_input, M))
    W_output_not_renorm[:, :, 0] += 1

    F_input = np.zeros((n_input, M))

    cred_cum_output = np.zeros(K_output)
    cred_cum_input = np.zeros((K_output, K_input))

    for m in range(M):
        obj_cur = L[m]

        # simulation of the input neurons
        input_neurons = np.zeros((n_input, N))
        cur_act = np.where(obj_cur == 1)[0]
        input_neurons[cur_act, :] = Poisson(freq, T, (cur_act.size, N))

        # simulation of the output neurons
        output_neurons = np.zeros((K_output, N))
        for j in range(K_output):
            probas = alpha[j] + np.sum(
                (
                    W_output[j, :n_input, m, nb][:, np.newaxis]
                    - W_output[j, n_input:, m, nb][:, np.newaxis]
                )
                * input_neurons,
                axis=0,
            )
            clipped_probas = np.clip(probas, 0, 1)
            output_neurons[j, :] = stats.bernoulli.rvs(clipped_probas)

        # firing rates
        F_input[:, m] = np.sum(input_neurons, axis=1) / T
        F_output[:, m, nb] = np.sum(output_neurons, axis=1) / T

        if (in_A(obj_cur) and F_output[0, m, nb] > F_output[1, m, nb]) or (
            in_B(obj_cur) and F_output[1, m, nb] > F_output[0, m, nb]
        ):
            answers[nb, m] = 1

        cred = cred_output_HAN(K_output, K_input, F_input[:, m], dt, obj_cur)
        cred_cum_output += np.sum(W_output[:, :, m, nb] * cred, axis=1)
        cred_cum_input += cred

        if m < M - 1:
            if expert == "EWA":
                W_output_not_renorm[:, :, m + 1] = EWA(
                    W_output_not_renorm[:, :, m], eta_output, cred, K_output
                )
            elif expert == "Multilin":
                W_output_not_renorm[:,:,m+1]=Multilin(W_output_not_renorm[:,:,m],eta_output,list_cred_output[:,:,m],K_output)
            elif expert == "PWA":
                W_output_not_renorm[:,:,m+1]=PWA(para_PWA,K_output,K_input,cred_cum_output,cred_cum_input)

            W_output[:, :, m + 1, nb] = (
                W_output_not_renorm[:, :, m + 1]
                / np.sum(W_output_not_renorm[:, :, m + 1], axis=1)[:, np.newaxis]
            )

        if m % 500 == 0:
            print(nb, m)

#np.save(f"{expert}100_inhib",answersEWA)
np.save(f"{expert}100W_inhib", W_output)
np.save(f"{expert}100F_inhib", F_output)
