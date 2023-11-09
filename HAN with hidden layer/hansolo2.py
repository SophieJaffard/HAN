import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


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

def list_obj_random(obj, M):
    """
    Create a matrix of random permutations of the input list 'obj' until it reaches
    the shape (M, len(obj)), where each row represents a block of randomly permuted objects. 
    Note: this function assumes that M is a multiple of n.
    """
    n = len(obj)
    num_blocks = int(M / n)
    
    # Create an array of random permutations for each block
    permutations = np.array([np.random.permutation(obj) for _ in range(num_blocks)])
    
    # Stack the permutations vertically to form the matrix
    result_matrix = np.vstack(permutations)
    
    return result_matrix

def list_objects_random(objects, M):
    """
    Returns a list of length `M` where each element is randomly chosen (with replacement)
    from the input list `objects`. Also returns a list of indices where each sublist
    contains the indices in the output list where the corresponding object from `objects` appears.
    """
    n = len(objects)
    random_indices = np.random.randint(n, size=M)
    L = np.array([objects[i] for i in random_indices])
    return L


def in_A(obj):
    return (obj[0] == 1 and obj[3] == 1) or (obj[1]==1 and obj[2]==1)


def in_B(objet):  # everything except the blue circle
    return not in_A(objet)


def simu_dist(p, size=1):
    """
    Vectorized version of simu_dist to generate multiple indices at once.
    """
    u = stats.uniform.rvs(0, 1, size=size)
    cumulative_probs = np.cumsum(p)
    return np.searchsorted(cumulative_probs, u)


def Poisson(p, shape):
    """
    Simulates a Poisson process over a time `T` and intensity `freq`.
    Returns a numpy array representing the Poisson process.
    """
    proc = stats.bernoulli.rvs(p, size=shape)
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

def Hawkes(V, W, nu):
    """
    Simulates a Hawkes process. Given the neighbors 'V', the weights 'W', and the bias "nu", the function
    simulates the process by computing the conditional spiking probability.
    """
    N = V.shape[1] #nb of points
    H = np.zeros(N) #contain points
    n_input = V.shape[0]
    probas = -nu + np.sum(
        (
            W[:n_input]-W[n_input:]
        )
        *V.T,
        axis=1
    )
    clipped_probas = np.clip(probas, 0, 1)
    H = stats.bernoulli.rvs(clipped_probas)
    return H


def Gain_Output(K_output, K_mid, P_mid, obj):  # new gains
    gain = np.zeros((K_output, K_mid))
    if in_A(obj):
        gain[0, :] = 2 * P_mid 
        gain[1, :] = -2 * P_mid
    else:
        gain[0, :] = -2 * P_mid
        gain[1, :] = 2 * P_mid
    return gain

def Gain_Mid(K_mid,K_input, input_neurons): #specific to ex
    gain = np.zeros((K_mid,K_input))
    N=input_neurons.shape[1]
    n_input=input_neurons.shape[0]

    gain[0, :n_input] = (1/N)*np.sum(input_neurons*input_neurons[0,:]*input_neurons[2,:], axis=1)
    gain[0, n_input:] = (1/N)*np.sum(input_neurons[0,:]*input_neurons[2,:]) - gain[0, :n_input]    

    gain[1, :n_input] = (1/N)*np.sum(input_neurons*input_neurons[0,:]*input_neurons[3,:], axis=1)
    gain[1, n_input:] = (1/N)*np.sum(input_neurons[0,:]*input_neurons[3,:]) - gain[1, :n_input]    

    gain[2, :n_input] = (1/N)*np.sum(input_neurons*input_neurons[1,:]*input_neurons[2,:], axis=1)
    gain[2, n_input:] = (1/N)*np.sum(input_neurons[1,:]*input_neurons[2,:]) - gain[2, :n_input]    

    gain[3, :n_input] = (1/N)*np.sum(input_neurons*input_neurons[1,:]*input_neurons[3,:], axis=1)
    gain[3, n_input:] = (1/N)*np.sum(input_neurons[1,:]*input_neurons[3,:]) - gain[3, :n_input]    

    return gain


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
