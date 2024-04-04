import numpy as np

coh = np.genfromtxt('COH.csv', delimiter=',')


with open('COH.npy', 'wb') as f:
    np.save(f, coh)


# with open('COH.npy', 'rb') as f:
#     coh = np.load(f)


print(coh)