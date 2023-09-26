# %% [markdown]
# Preliminary functions

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


## SV: this is builtin in numpy
## SV: old style code is `np.random.permutation`
## SV: and it is better to use a `np.random.Generator`
# def objects_random(obj):
#     n=len(obj)
#     a=np.arange(n)
#     np.random.shuffle(a)
#     obj_random=[obj[a[i]] for i in range(n)]
#     return obj_random
def objects_random(obj):
    return list(np.random.permutation(obj))


## SV: instead of a single comment, use a doctring
## SV: see https://peps.python.org/pep-0257/
# def list_obj(obj,M): #obj=list objects, return alternated list of objects
#     n=len(obj)
#     ind=[[n*k+i for k in range(int(M/n)) ] for i in range(n) ]
#     li=[obj[i%n] for i in range(M)]
#     return li, ind
def list_obj(obj, M):
    """
    Create an alternated list of objects from the input list `obj` until it reaches the length `M`.
    Also returns a list of indices representing the pattern of the alternated list.
    Note: This function assumes that M is a multiple of n.
    """
    n = len(obj)
    ind = np.tile(np.arange(n), (int(M/n), 1)).T
    li = [obj[i % n] for i in range(M)]
    return li, ind


## SV: always try to use builtin
# def list_objects_random(objects,M): #returns list with random objects (tirés avec remise unif)
#     L=[]
#     n=len(objects)
#     ind=[]
#     for j in range(n):
#         ind.append([])
#     for k in range(M):
#         i=np.random.randint(n)
#         L.append(objects[i])
#         ind[i].append(k)
#     return L, ind
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

## SV: The structure if BLABLA then True else False is "code smell"
## SV: You can just return the expression
# def in_B(objet):  # exception = blue circle 100100
#     if objet[0] == 1 and objet[3] == 1:
#         return True
#     return False
def in_B(obj):
    return obj[0] == 1 and obj[3] == 1

def in_A(objet):  # everything except the blue circle
    return not in_B(objet)

## SV: Use np.cumsum and np.searchsorted
# def simu_dist(p):  # simulation of the probability distribution p (finite vector, gives proba of 0,...,n-1)
#     n = np.size(p)
#     u = stats.uniform.rvs(0, 1)
#     somme = p[0]
#     for i in range(n-1):
#         if u <= somme:
#             return i
#         somme += p[i+1]
#     return n-1
def simu_dist(p, size=1):
    """
    Vectorized version of simu_dist to generate multiple indices at once.
    """
    u = stats.uniform.rvs(0, 1, size=size)
    cumulative_probs = np.cumsum(p)
    return np.searchsorted(cumulative_probs, u)

## SV: try to always avoid loop. corollary: use vectorized cod
## SV: try to avoid variable name starting with an uppercase
## SV: i.e. P -> p
# def Poisson(freq, N, T):  # freq = intensity, N=nb of steps, T=simulation time
#     P = np.zeros(N)
#     dt = T/N
#     for i in range(N):
#         P[i] = stats.bernoulli.rvs(freq*dt)
#     return P
def Poisson(freq, T, shape):
    """
    Simulates a Poisson process over a time `T` and intensity `freq`.
    Returns a numpy array representing the Poisson process.
    """
    dt = T/N
    proc = stats.bernoulli.rvs(freq*dt, size=shape)
    return proc

## SV: dont use np.size, use V.shape instead
## SV: using searchsorted, you do not need to convert to int
# def Hawkes_lin(V, W):  # W=weights, V=neighboors, no spontaneous rate, using Kalikow
#     N = np.size(V[0, :])  # nb of points
#     H = np.zeros(N)  # contain points
#     for i in range(N):
#         ind_v = simu_dist(W)
#         H[i] = V[int(ind_v), i]
#     return H
# def Hawkes_lin(V, W):
#     """
#     Simulates a Hawkes process (no spontaneous rate) using the Kalikow decomposition.
#     Given the neighbors `V` and the weights `W`, the function simulates the process 
#     by selecting points from the neighbors based on the weights.
#     """
#     n = V.shape[1]  # nb of points
#     H = np.zeros(n)  # contain points
#     for i in range(n):
#         ind_v = simu_dist(W)
#         H[i] = V[int(ind_v), i]
#     return H
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

def cred_output_HAN_Solo(K_output, K_input, F_input, dt,obj): #new gains
    cred=np.zeros((K_output,K_input))
    if in_A(obj):
        cred[0,:]=(9/8)*F_input*dt
        cred[1,:]=-(9/8)*F_input*dt
    else:
        cred[0,:]=-9*F_input*dt
        cred[1,:]=9*F_input*dt
    return cred

def cred_output_HAN(K_output, K_input, F_input, dt,obj): #new gains
    cred=np.zeros((K_output,K_input))
    n_input=int(K_input/2)
    if in_A(obj):
        cred[0,:n_input]=(9/8)*F_input*dt
        cred[1,:n_input]=-(9/8)*F_input*dt
        cred[0,n_input:]=-(9/8)*F_input*dt
        cred[1,n_input:]=(9/8)*F_input*dt
    else:
        cred[0,:n_input]=-9*F_input*dt
        cred[1,:n_input]=9*F_input*dt
        cred[0,n_input:]=9*F_input*dt
        cred[1,n_input:]=-9*F_input*dt
    return cred

## SV: broadcasting
## SV: why do we want to stop at K_outpu??!  
# def EWA(W_not_renorm,eta,cred,K_output): #return next W_not_renorm using EWA
#     res=W_not_renorm
#     for j in range(K_output):
#         res[j,:]=res[j,:]*np.exp(eta*cred[j,:])
#     return res
def EWA(W_not_renorm, eta, cred, K_output):
    """
    Updates the weights `W_not_renorm` using the Exponentially Weighted Average (EWA) method.
    """
    res = W_not_renorm.copy()
    res[:K_output, :] *= np.exp(eta * cred[:K_output, :])
    return res

## SV: same
# def Multilin(W_not_renorm,eta,cred,K_output): #return next W_not_renorm using Multilin
#     res=W_not_renorm
#     for j in range(K_output):
#         res[j,:]=res[j,:]*(1+eta*cred[j,:])
#     return res
def Multilin(W_not_renorm, eta, cred, K_output):
    """
    Updates the weights `W_not_renorm` using a multiplicative linear update.
    """
    res = W_not_renorm.copy()
    res[:K_output, :] *= (1 + eta * cred[:K_output, :])
    return res

## SV: same
# def PWA(p,K_output,K_input,cred_cum_output, cred_cum_input): 
#     res=np.zeros((K_output,K_input))
#     for j in range(K_output):
#         for i in range(K_input):
#             res[j,i] = (np.maximum(0,cred_cum_input[j,i]-cred_cum_output[j]))**(p-1)
#     return res
def PWA(p, K_output, K_input, cred_cum_output, cred_cum_input):
    """
    Updates the weights `W_not_renorm` using the PWA method.    
    """
    diff = np.maximum(0, cred_cum_input[:K_output, :K_input] - cred_cum_output[:K_output, np.newaxis])
    res = diff**(p-1)
    return res

# %% [markdown]
# HAN Solo (linear case)

# %%
#np.random.seed(10)

Nb=100 

M=2997
dt=0.002
N=1000
T=N*dt

freq=100 #firing rate input neurons +
freq2=150 #firing rate input neurons -

p=freq*dt #spiking proba input neurons
p2=freq2*dt

K_input=12 #nb input neurons
K_output=2 #nb output neurons

sup = freq2*dt*((9/8)+9)
eta_output=np.sqrt(8*np.log(K_input)/M)/sup #para EWA


obj=[[1,0,0,1,0,0],[1,0,0,0,1,0],[1,0,0,0,0,1],[0,1,0,1,0,0],[0,1,0,0,1,0],[0,1,0,0,0,1],[0,0,1,1,0,0],[0,0,1,0,1,0],[0,0,1,0,0,1]]
 #blue circle blue square blue triangle red circle red square red triangle gray circle gray square gray triangle 

obj_tot=[] #matrix of objects of each realisation

activity=[] #contains active neurons for corresponding object
answers=np.zeros((Nb,M)) #0 if wrong, 1 if correct

W_output=np.zeros((K_output, K_input,M,Nb)) #weights output neurons
W_output[:,:,0,:]+=1/K_input #initialisation
F_output=np.zeros((K_output, M,Nb)) #firing rate output neurons

#para_PWA=2*np.log(K_input) para opti PWA for regret
para_PWA=2

## SV: use list comprehension
#index of active neurons for each object
activity = [o+[1-l for l in o] for o in obj]


for nb in range(Nb):
    act2=objects_random(activity) #list of active neurons for this realisation
    obj_tot.append(act2[:int(K_input/2)]) 
    L,ind=list_obj(act2,M) #list and indexes of objects

    W_output_not_renorm=np.zeros((K_output, K_input,M)) 
    W_output_not_renorm[:,:,0]+=1

    F_input=np.zeros((K_input,M)) #firing rate input neurons
    

    cred_cum_output=np.zeros(K_output) 
    cred_cum_input=np.zeros((K_output,K_input))


    for m in range(M):
        activity_cur=L[m] #current active neurons
        obj_cur=activity_cur[:int(K_input/2)] #current object

        #simulation of the input neurons
        ## SV: let's vectorize!
        # for i in range(K_input):
        #     if activity_cur[i]==1:
        #         if i < int(K_input/2):
        #             input_neurons[i,:]=Poisson(freq,N,T)
        #         else:
        #             input_neurons[i,:]=Poisson(freq2,N,T)
        input_neurons=np.zeros((K_input, N))
        cur_act = np.where(activity_cur == 1)[0]
        first_split = cur_act[cur_act < (K_input // 2)]
        second_split = cur_act[cur_act >= (K_input // 2)]
        input_neurons[first_split,:] = Poisson(freq, T, (first_split.size, N))
        input_neurons[second_split,:] = Poisson(freq2, T, (second_split.size, N))

        #simulation of the output neurons
        ## SV: note -> harder to vectorize :( Doable?
        output_neurons=np.zeros((K_output,N))
        for i in range(K_output):
            output_neurons[i,:]= Hawkes_lin(input_neurons,W_output[i,:,m,nb])
    
        #firing rates
        ## SV: vectorize
        # for i in range(K_input):
        #     F_input[i,m]=np.sum(input_neurons[i,:])/T 
        F_input[:, m] = np.sum(input_neurons, axis=1) / T
    
        ## SV: vectorize
        # for i in range(K_output):
        #     F_output[i,m,nb]=np.sum(output_neurons[i,:])/T 
        F_output[:, m, nb] = np.sum(output_neurons, axis=1) / T
        
        if (in_A(obj_cur) and F_output[0,m,nb] > F_output[1,m,nb]) or (in_B(obj_cur) and F_output[1,m,nb] > F_output[0,m,nb]):
            answers[nb,m]=1
    
        cred=cred_output_HAN_Solo(K_output, K_input, F_input[:,m], dt,obj_cur)
        ## SV: vectorize
        # for j in range(K_output):
        #     cred_cum_input[j,:]+=cred[j,:]
        #     cred_cum_output[j]+=np.sum(W_output[j,:,m,nb]* cred[j,:])
        cred_cum_input += cred
        cred_cum_output += np.sum(W_output[:, :, m, nb] * cred, axis=1)

        if m<M-1 :
            W_output_not_renorm[:,:,m+1]=EWA(W_output_not_renorm[:,:,m],eta_output,cred,K_output)
            #W_output_not_renorm[:,:,m+1]=PWA(para_PWA,K_output,K_input,cred_cum_output,cred_cum_input)

            for j in range(K_output):
                W_output[j,:,m+1,nb]=W_output_not_renorm[j,:,m+1]/np.sum(W_output_not_renorm[j,:,m+1])

        if m%100==0:
            print(nb,m)

#np.save('PWA100_para_opti',answersEWA)
#np.save('PWA100W_para_opti', W_output)
#np.save('PWA100F_para_opti', F_output)

# %% [markdown]
# HAN (nonlinear case)

# %%
Nb=100

M=2997
dt=0.002
N=1000
T=N*dt


freq=100 #firing rate input neurons +
freq2=150

alpha=np.array([100*dt,0]) #spontaneous rates of output neurons

p=freq*dt #spiking proba input neurons
p2=freq2*dt

K_input=12 #nb experts
n_input=6 #nb input neurons
K_output=2 #nb output neurons

sup = freq2*dt*((9/8)+9)
eta_output=np.sqrt(8*np.log(K_input)/M)/sup #para EWA


obj=[[1,0,0,1,0,0],[1,0,0,0,1,0],[1,0,0,0,0,1],[0,1,0,1,0,0],[0,1,0,0,1,0],[0,1,0,0,0,1],[0,0,1,1,0,0],[0,0,1,0,1,0],[0,0,1,0,0,1]]
 #blue circle blue square blue triangle red circle red square red triangle gray circle gray square gray triangle 

obj_tot=[]

answers=np.zeros((Nb,M)) #0 if wrong, 1 if correct

W_output=np.zeros((K_output, K_input,M,Nb)) 
W_output[:,:,0,:]+=1/K_input
F_output=np.zeros((K_output, M,Nb)) #firing rate output neurons

#para_PWA=2*np.log(K_input)
para_PWA=2


for nb in range(Nb):
    obj2=objects_random(obj)
    obj_tot.append(obj2)
    L,ind=list_obj(obj2,M)

    W_output_not_renorm=np.zeros((K_output, K_input,M)) 
    W_output_not_renorm[:,:,0]+=1

    F_input=np.zeros((n_input,M))
    
    

    cred_cum_output=np.zeros(K_output)
    cred_cum_input=np.zeros((K_output,K_input))


    for m in range(M):
        
        obj_cur=L[m]

        #simulation of the input neurons
        
        # input_neurons=np.zeros((n_input, N))
        # for i in range(n_input):
        #     if obj_cur[i]==1:
        #        input_neurons[i,:]=Poisson(freq,N,T)
        input_neurons=np.zeros((n_input, N))
        cur_act = np.where(obj_cur == 1)[0]
        input_neurons[cur_act,:] = Poisson(freq, T, (cur_act.size, N))
                
        #simulation of the output neurons
        output_neurons=np.zeros((K_output,N))
        for j in range(K_output):
            # for t in range(N):
            #     ## SV: you should use np.clip, it's faster
            #     # output_neurons[j,t]= stats.bernoulli.rvs(np.minimum(np.maximum(0,alpha[j]+np.sum((W_output[j,:n_input,m,nb] - W_output[j,n_input:,m,nb])*input_neurons[:,t])),1))
            #     output_neurons[j,t]= stats.bernoulli.rvs(np.clip(alpha[j]+np.sum((W_output[j,:n_input,m,nb] - W_output[j,n_input:,m,nb])*input_neurons[:,t]),0,1))
            probas = alpha[j] + np.sum((W_output[j, :n_input, m, nb][:, np.newaxis] - W_output[j, n_input:, m, nb][:, np.newaxis]) * input_neurons, axis=0)
            clipped_probas = np.clip(probas, 0, 1)
            output_neurons[j, :] = stats.bernoulli.rvs(clipped_probas)
    
        #firing rates
        # for i in range(n_input):
        #     F_input[i,m]=np.sum(input_neurons[i,:])/T 
        F_input[:, m] = np.sum(input_neurons, axis=1) / T
    
        # for i in range(K_output):
        #     F_output[i,m,nb]=np.sum(output_neurons[i,:])/T 
        F_output[:, m, nb] = np.sum(output_neurons, axis=1) / T
        
        if (in_A(obj_cur) and F_output[0,m,nb] > F_output[1,m,nb]) or (in_B(obj_cur) and F_output[1,m,nb] > F_output[0,m,nb]):
            answers[nb,m]=1
    
        cred=cred_output_HAN(K_output, K_input, F_input[:,m], dt,obj_cur)
        # for j in range(K_output):
        #     cred_cum_output[j]+=np.sum(W_output[j,:,m,nb]*cred[j,:])
        #     cred_cum_input[j,:]+=cred[j,:]
        cred_cum_output += np.sum(W_output[:, :, m, nb] * cred, axis=1)
        cred_cum_input += cred

        if m<M-1 :
            W_output_not_renorm[:,:,m+1]=EWA(W_output_not_renorm[:,:,m],eta_output,cred,K_output)
            #W_output_not_renorm[:,:,m+1]=Multilin(W_output_not_renorm[:,:,m],eta_output,list_cred_output[:,:,m],K_output)
            #W_output_not_renorm[:,:,m+1]=PWA(para_PWA,K_output,K_input,cred_cum_output,cred_cum_input)

            # for j in range(K_output):
            #     W_output[j,:,m+1,nb]=W_output_not_renorm[j,:,m+1]/np.sum(W_output_not_renorm[j,:,m+1])
            W_output[:, :, m+1, nb] = W_output_not_renorm[:, :, m+1] / np.sum(W_output_not_renorm[:, :, m+1], axis=1)[:, np.newaxis]

        if m%100==0:
            print(nb,m)

#np.save('PWA100_inhib',answersEWA)
#np.save('PWA100W_inhib', W_output)
#np.save('PWA100F_inhib', F_output)

# %% [markdown]
# One realisation (pas bonne légende pour les objets, à changer)

# %%
nb=0

## SV: Try to not unroll loop when you do not have good reason to do it
# fig, axs=plt.subplots(3,3, sharex=True, sharey=True)

# axs[0,0].scatter(ind[0], F_output[0,ind[0],nb],label="$A$",s=5)
# axs[0,0].scatter(ind[0], F_output[1,ind[0],nb],label="$B$",s=5)
# axs[0,0].set_title("Blue circle")


# axs[0,1].scatter(ind[1], F_output[0,ind[1],nb],label="$A$",s=5)
# axs[0,1].scatter(ind[1], F_output[1,ind[1],nb],label="$B$",s=5)
# axs[0,1].set_title("Blue square")


# axs[0,2].scatter(ind[2], F_output[0,ind[2],nb],label="$A$",s=5)
# axs[0,2].scatter(ind[2], F_output[1,ind[2],nb],label="$B$",s=5)
# axs[0,2].set_title("Blue triangle")


# axs[1,0].scatter(ind[3], F_output[0,ind[3],nb],label="$A$",s=5)
# axs[1,0].scatter(ind[3], F_output[1,ind[3],nb],label="$B$",s=5)
# axs[1, 0].set_ylabel("Empirical firing rates")
# axs[1,0].set_title("Red circle")


# axs[1,1].scatter(ind[4], F_output[0,ind[4],nb],label="$A$",s=5)
# axs[1,1].scatter(ind[4], F_output[1,ind[4],nb],label="$B$",s=5)
# axs[1,1].set_title("Red square")


# axs[1,2].scatter(ind[5], F_output[0,ind[5],nb],label="$A$",s=5)
# axs[1,2].scatter(ind[5], F_output[1,ind[5],nb],label="$B$",s=5)
# axs[1,2].set_title("Red triangle")


# axs[2,0].scatter(ind[6], F_output[0,ind[6],nb],label="$A$",s=5)
# axs[2,0].scatter(ind[6], F_output[1,ind[6],nb],label="$B$",s=5)
# axs[2,0].set_title("Gray circle")


# axs[2,1].scatter(ind[7], F_output[0,ind[7],nb],label="$A$",s=5)
# axs[2,1].scatter(ind[7], F_output[1,ind[7],nb],label="$B$",s=5)
# axs[2,1].set_title("Gray square")
# axs[2, 1].set_xlabel("m")

# axs[2,2].scatter(ind[8], F_output[0,ind[8],nb],label="$A$",s=5)
# axs[2,2].scatter(ind[8], F_output[1,ind[8],nb],label="$B$",s=5)
# axs[2,2].set_title("Gray triangle")
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)

titles = ["Blue circle", "Blue square", "Blue triangle",
          "Red circle", "Red square", "Red triangle",
          "Gray circle", "Gray square", "Gray triangle"]

for i in range(3):
    for j in range(3):
        idx = i * 3 + j
        axs[i, j].scatter(ind[idx], F_output[0, ind[idx], nb], label="$A$", s=5)
        axs[i, j].scatter(ind[idx], F_output[1, ind[idx], nb], label="$B$", s=5)
        axs[i, j].set_title(titles[idx])

axs[1, 0].set_ylabel("Empirical firing rates")
axs[2, 1].set_xlabel("m")

plt.legend()
fig.tight_layout()
#plt.savefig('fr_AB.pdf')
plt.show()
#print(obj_tot)

# %%
##Figure HAN et HAN Solo

answersEWA=np.load('EWA100.npy')
M=2997
Nb=100
paq=180

level=0.1

plt.figure(figsize=(5.5,4))

answersEWA2=np.load('EWA100_inhib.npy')
M=2997
Nb=100
paq=180

level=0.1

Y=np.zeros((Nb,M-paq)) # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M-paq):
        Y[nb,m]=np.sum(answersEWA2[nb,m:m+paq])/paq


Y_med=np.zeros(M-paq)
Y_90=np.zeros(M-paq)
Y_10=np.zeros(M-paq)
Y_max=np.zeros(M-paq)

for m in range(M-paq):
    X=Y[:,m]
    X=np.sort(X)
    Y_med[m]=X[int(Nb/2)]
    Y_90[m]=X[int(Nb*(1-level))]
    Y_10[m]=X[int(Nb*level)]
    Y_max[m]=X[Nb-1]
   # if Y_10[m] > Y_med[m]:
        #print(m)

X2=np.arange(M-paq)+paq
plt.plot(X2,Y_med, label='HAN with EWA', color='g')
plt.fill_between(X2, Y_10, Y_90, color='g', alpha=.2)



answersPWA2=np.load('PWA100_inhib.npy') 
#answersPWA=np.load('PWA100_para_opti.npy') 

Y=np.zeros((Nb,M-paq)) # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M-paq):
        Y[nb,m]=np.sum(answersPWA2[nb,m:m+paq])/paq


Y_med=np.zeros(M-paq)
Y_90=np.zeros(M-paq)
Y_10=np.zeros(M-paq)


for m in range(M-paq):
    X=Y[:,m]
    X=np.sort(X)
    Y_med[m]=X[int(Nb/2)]
    Y_90[m]=X[int(Nb*(1-level))]
    Y_10[m]=X[int(Nb*level)]
    Y_max[m]=X[Nb-1]
   # if Y_10[m] > Y_med[m]:
        #print(m)

X2=np.arange(M-paq)+paq

plt.plot(X2,Y_med, label='HAN with PWA',color='magenta')

plt.fill_between(X2, Y_10, Y_90, color='magenta', alpha=.1)



Y=np.zeros((Nb,M-paq)) # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M-paq):
        Y[nb,m]=np.sum(answersEWA[nb,m:m+paq])/paq


Y_med=np.zeros(M-paq)
Y_90=np.zeros(M-paq)
Y_10=np.zeros(M-paq)
Y_max=np.zeros(M-paq)

for m in range(M-paq):
    X=Y[:,m]
    X=np.sort(X)
    Y_med[m]=X[int(Nb/2)]
    Y_90[m]=X[int(Nb*(1-level))]
    Y_10[m]=X[int(Nb*level)]
    Y_max[m]=X[Nb-1]
   # if Y_10[m] > Y_med[m]:
        #print(m)

X2=np.arange(M-paq)+paq
plt.plot(X2,Y_med, label='HAN Solo with EWA',color='b')
plt.fill_between(X2, Y_10, Y_90, color='b', alpha=.1)



answersPWA=np.load('PWA100.npy') 
#answersPWA=np.load('PWA100_para_opti.npy') 

Y=np.zeros((Nb,M-paq)) # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M-paq):
        Y[nb,m]=np.sum(answersPWA[nb,m:m+paq])/paq


Y_med=np.zeros(M-paq)
Y_90=np.zeros(M-paq)
Y_10=np.zeros(M-paq)


for m in range(M-paq):
    X=Y[:,m]
    X=np.sort(X)
    Y_med[m]=X[int(Nb/2)]
    Y_90[m]=X[int(Nb*(1-level))]
    Y_10[m]=X[int(Nb*level)]
    Y_max[m]=X[Nb-1]
   # if Y_10[m] > Y_med[m]:
        #print(m)

X2=np.arange(M-paq)+paq

plt.plot(X2,Y_med, label='HAN Solo with PWA',color='orange')

plt.fill_between(X2, Y_10, Y_90, color='orange', alpha=.2)



plt.yticks([0.6,0.8,1])
plt.xticks([500,1500,3000])
plt.ylabel('Proportion of correction classifications')
plt.xlabel('m')

plt.legend()
plt.savefig('curves.pdf')
plt.show()




# %%
##Figure HAN et HAN Solo et CC

answersEWA=np.load('EWA100.npy')
M=2997
Nb=100
paq=180

level=0.1

plt.figure(figsize=(5.5,4))

answersEWA2=np.load('EWA100_inhib.npy')
M=2997
Nb=100
paq=180

level=0.1

Y=np.zeros((Nb,M-paq)) # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M-paq):
        Y[nb,m]=np.sum(answersEWA2[nb,m:m+paq])/paq


Y_med=np.zeros(M-paq)
Y_90=np.zeros(M-paq)
Y_10=np.zeros(M-paq)
Y_max=np.zeros(M-paq)

for m in range(M-paq):
    X=Y[:,m]
    X=np.sort(X)
    Y_med[m]=X[int(Nb/2)]
    Y_90[m]=X[int(Nb*(1-level))]
    Y_10[m]=X[int(Nb*level)]
    Y_max[m]=X[Nb-1]
   # if Y_10[m] > Y_med[m]:
        #print(m)

X2=np.arange(M-paq)+paq
plt.plot(X2,Y_med, label='HAN with EWA', color='g')
plt.fill_between(X2, Y_10, Y_90, color='g', alpha=.2)



answersPWA2=np.load('PWA100_inhib.npy') 
#answersPWA=np.load('PWA100_para_opti.npy') 

Y=np.zeros((Nb,M-paq)) # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M-paq):
        Y[nb,m]=np.sum(answersPWA2[nb,m:m+paq])/paq


Y_med=np.zeros(M-paq)
Y_90=np.zeros(M-paq)
Y_10=np.zeros(M-paq)


for m in range(M-paq):
    X=Y[:,m]
    X=np.sort(X)
    Y_med[m]=X[int(Nb/2)]
    Y_90[m]=X[int(Nb*(1-level))]
    Y_10[m]=X[int(Nb*level)]
    Y_max[m]=X[Nb-1]
   # if Y_10[m] > Y_med[m]:
        #print(m)

X2=np.arange(M-paq)+paq

plt.plot(X2,Y_med, label='HAN with PWA',color='magenta')

plt.fill_between(X2, Y_10, Y_90, color='magenta', alpha=.1)



Y=np.zeros((Nb,M-paq)) # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M-paq):
        Y[nb,m]=np.sum(answersEWA[nb,m:m+paq])/paq


Y_med=np.zeros(M-paq)
Y_90=np.zeros(M-paq)
Y_10=np.zeros(M-paq)
Y_max=np.zeros(M-paq)

for m in range(M-paq):
    X=Y[:,m]
    X=np.sort(X)
    Y_med[m]=X[int(Nb/2)]
    Y_90[m]=X[int(Nb*(1-level))]
    Y_10[m]=X[int(Nb*level)]
    Y_max[m]=X[Nb-1]
   # if Y_10[m] > Y_med[m]:
        #print(m)

X2=np.arange(M-paq)+paq
plt.plot(X2,Y_med, label='HAN Solo with EWA',color='b')
plt.fill_between(X2, Y_10, Y_90, color='b', alpha=.1)



answersPWA=np.load('PWA100.npy') 
#answersPWA=np.load('PWA100_para_opti.npy') 

Y=np.zeros((Nb,M-paq)) # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M-paq):
        Y[nb,m]=np.sum(answersPWA[nb,m:m+paq])/paq


Y_med=np.zeros(M-paq)
Y_90=np.zeros(M-paq)
Y_10=np.zeros(M-paq)


for m in range(M-paq):
    X=Y[:,m]
    X=np.sort(X)
    Y_med[m]=X[int(Nb/2)]
    Y_90[m]=X[int(Nb*(1-level))]
    Y_10[m]=X[int(Nb*level)]
    Y_max[m]=X[Nb-1]
   # if Y_10[m] > Y_med[m]:
        #print(m)

X2=np.arange(M-paq)+paq

plt.plot(X2,Y_med, label='HAN Solo with PWA',color='orange')

plt.fill_between(X2, Y_10, Y_90, color='orange', alpha=.2)

answersCC=np.load('CC.npy') 
#answersPWA=np.load('PWA100_para_opti.npy') 

Y=np.zeros((Nb,M-paq)) # percentage "glissant" of correct guesses for a bloc of 18 objects
for nb in range(Nb):
    for m in range(M-paq):
        Y[nb,m]=np.sum(answersCC[nb,m:m+paq])/paq


Y_med=np.zeros(M-paq)
Y_90=np.zeros(M-paq)
Y_10=np.zeros(M-paq)


for m in range(M-paq):
    X=Y[:,m]
    X=np.sort(X)
    Y_med[m]=X[int(Nb/2)]
    Y_90[m]=X[int(Nb*(1-level))]
    Y_10[m]=X[int(Nb*level)]
    Y_max[m]=X[Nb-1]
   # if Y_10[m] > Y_med[m]:
        #print(m)

X2=np.arange(M-paq)+paq

plt.plot(X2,Y_med, label='Component Cue',color='brown')

plt.fill_between(X2, Y_10, Y_90, color='brown', alpha=.2)



plt.yticks([0.6,0.8,1])
plt.xticks([500,1500,3000])
plt.ylabel('Proportion of correction classifications')
plt.xlabel('m')

plt.legend()
plt.savefig('curves.pdf')
plt.show()





