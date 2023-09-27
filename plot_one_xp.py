# One realisation (pas bonne légende pour les objets, à changer)
## TODO: fix it
import numpy as np
import matplotlib.pyplot as plt
from hansolo import *

nb = 0

fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)

titles = [
    "Blue circle",
    "Blue square",
    "Blue triangle",
    "Red circle",
    "Red square",
    "Red triangle",
    "Gray circle",
    "Gray square",
    "Gray triangle",
]

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
# plt.savefig('fr_AB.pdf')
plt.show()
# print(obj_tot)
