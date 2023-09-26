import numpy as np
import scipy.stats as stats
from hansolo import *

# np.random.seed(10)

Nb = 100

M = 2997
dt = 0.002
N = 1000
T = N * dt

freq = 100  # firing rate input neurons +
freq2 = 150  # firing rate input neurons -

p = freq * dt  # spiking proba input neurons
p2 = freq2 * dt

K_input = 12  # nb input neurons
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

obj_tot = []  # matrix of objects of each realisation

activity = []  # contains active neurons for corresponding object
answers = np.zeros((Nb, M))  # 0 if wrong, 1 if correct

W_output = np.zeros((K_output, K_input, M, Nb))  # weights output neurons
W_output[:, :, 0, :] += 1 / K_input  # initialisation
F_output = np.zeros((K_output, M, Nb))  # firing rate output neurons

# para_PWA=2*np.log(K_input) para opti PWA for regret
para_PWA = 2

## SV: use list comprehension
# index of active neurons for each object
activity = [o + [1 - l for l in o] for o in obj]


for nb in range(Nb):
    act2 = objects_random(activity)  # list of active neurons for this realisation
    obj_tot.append(act2[: int(K_input / 2)])
    L, ind = list_obj(act2, M)  # list and indexes of objects

    W_output_not_renorm = np.zeros((K_output, K_input, M))
    W_output_not_renorm[:, :, 0] += 1

    F_input = np.zeros((K_input, M))  # firing rate input neurons

    cred_cum_output = np.zeros(K_output)
    cred_cum_input = np.zeros((K_output, K_input))

    for m in range(M):
        activity_cur = L[m]  # current active neurons
        obj_cur = activity_cur[: int(K_input / 2)]  # current object

        # simulation of the input neurons
        input_neurons = np.zeros((K_input, N))
        cur_act = np.where(activity_cur == 1)[0]
        first_split = cur_act[cur_act < (K_input // 2)]
        second_split = cur_act[cur_act >= (K_input // 2)]
        input_neurons[first_split, :] = Poisson(freq, T, (first_split.size, N))
        input_neurons[second_split, :] = Poisson(freq2, T, (second_split.size, N))

        # simulation of the output neurons
        ## SV: note -> harder to vectorize :( Doable?
        output_neurons = np.zeros((K_output, N))
        for i in range(K_output):
            output_neurons[i, :] = Hawkes_lin(input_neurons, W_output[i, :, m, nb])

        # firing rates
        F_input[:, m] = np.sum(input_neurons, axis=1) / T

        F_output[:, m, nb] = np.sum(output_neurons, axis=1) / T

        if (in_A(obj_cur) and F_output[0, m, nb] > F_output[1, m, nb]) or (
            in_B(obj_cur) and F_output[1, m, nb] > F_output[0, m, nb]
        ):
            answers[nb, m] = 1

        cred = cred_output_HAN_Solo(K_output, K_input, F_input[:, m], dt, obj_cur)
        cred_cum_input += cred
        cred_cum_output += np.sum(W_output[:, :, m, nb] * cred, axis=1)

        if m < M - 1:
            W_output_not_renorm[:, :, m + 1] = EWA(
                W_output_not_renorm[:, :, m], eta_output, cred, K_output
            )
            # W_output_not_renorm[:,:,m+1]=PWA(para_PWA,K_output,K_input,cred_cum_output,cred_cum_input)

            for j in range(K_output):
                W_output[j, :, m + 1, nb] = W_output_not_renorm[j, :, m + 1] / np.sum(
                    W_output_not_renorm[j, :, m + 1]
                )

        if m % 100 == 0:
            print(nb, m)

# np.save('PWA100_para_opti',answersEWA)
# np.save('PWA100W_para_opti', W_output)
# np.save('PWA100F_para_opti', F_output)
