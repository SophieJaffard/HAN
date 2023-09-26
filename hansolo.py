import numpy as np
import scipy.stats as stats


def objects_random(obj):
    return list(np.random.permutation(obj))


def list_obj(obj, M):
    """
    Create an alternated list of objects from the input list `obj` until it reaches the length `M`.
    Also returns a list of indices representing the pattern of the alternated list.
    Note: This function assumes that M is a multiple of n.
    """
    n = len(obj)
    ind = np.tile(np.arange(n), (int(M / n), 1)).T
    li = [obj[i % n] for i in range(M)]
    return li, ind


def list_objects_random(objects, M):
    """
    Returns a list of length `M` where each element is randomly chosen (with replacement)
    from the input list `objects`. Also returns a list of indices where each sublist
    contains the indices in the output list where the corresponding object from `objects` appears.
    """
    n = len(objects)
    random_indices = np.random.randint(n, size=M)
    L = [objects[i] for i in random_indices]

    ind = [list(np.where(random_indices == i)[0]) for i in range(n)]

    return L, ind


def in_B(obj):
    return obj[0] == 1 and obj[3] == 1


def in_A(objet):  # everything except the blue circle
    return not in_B(objet)


def simu_dist(p, size=1):
    """
    Vectorized version of simu_dist to generate multiple indices at once.
    """
    u = stats.uniform.rvs(0, 1, size=size)
    cumulative_probs = np.cumsum(p)
    return np.searchsorted(cumulative_probs, u)


def Poisson(freq, T, shape):
    """
    Simulates a Poisson process over a time `T` and intensity `freq`.
    Returns a numpy array representing the Poisson process.
    """
    dt = T / shape[-1]
    proc = stats.bernoulli.rvs(freq * dt, size=shape)
    return proc


def Hawkes_lin(V, W):
    """
    Simulates a Hawkes process (no spontaneous rate) using the Kalikow decomposition.
    Given the neighbors `V` and the weights `W`, the function simulates the process
    by selecting points from the neighbors based on the weights.
    """
    n = V.shape[1]  # nb of points
    H = np.zeros(n)  # contain points
    ind_v = simu_dist(W, n)
    H = V[ind_v, np.arange(n)]
    return H


def cred_output_HAN_Solo(K_output, K_input, F_input, dt, obj):  # new gains
    cred = np.zeros((K_output, K_input))
    if in_A(obj):
        cred[0, :] = (9 / 8) * F_input * dt
        cred[1, :] = -(9 / 8) * F_input * dt
    else:
        cred[0, :] = -9 * F_input * dt
        cred[1, :] = 9 * F_input * dt
    return cred


def cred_output_HAN(K_output, K_input, F_input, dt, obj):  # new gains
    cred = np.zeros((K_output, K_input))
    n_input = int(K_input / 2)
    if in_A(obj):
        cred[0, :n_input] = (9 / 8) * F_input * dt
        cred[1, :n_input] = -(9 / 8) * F_input * dt
        cred[0, n_input:] = -(9 / 8) * F_input * dt
        cred[1, n_input:] = (9 / 8) * F_input * dt
    else:
        cred[0, :n_input] = -9 * F_input * dt
        cred[1, :n_input] = 9 * F_input * dt
        cred[0, n_input:] = 9 * F_input * dt
        cred[1, n_input:] = -9 * F_input * dt
    return cred


def EWA(W_not_renorm, eta, cred, K_output):
    """
    Updates the weights `W_not_renorm` using the Exponentially Weighted Average (EWA) method.
    """
    res = W_not_renorm.copy()
    res[:K_output, :] *= np.exp(eta * cred[:K_output, :])
    return res


def Multilin(W_not_renorm, eta, cred, K_output):
    """
    Updates the weights `W_not_renorm` using a multiplicative linear update.
    """
    res = W_not_renorm.copy()
    res[:K_output, :] *= 1 + eta * cred[:K_output, :]
    return res


def PWA(p, K_output, K_input, cred_cum_output, cred_cum_input):
    """
    Updates the weights `W_not_renorm` using the PWA method.
    """
    diff = np.maximum(
        0, cred_cum_input[:K_output, :K_input] - cred_cum_output[:K_output, np.newaxis]
    )
    res = diff ** (p - 1)
    return res
