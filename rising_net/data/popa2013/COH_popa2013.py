import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


ftarg = np.arange(5.0, 48.0, 1.0)


def interpolate(data, f=ftarg):
    # Compute coherence interpolation...
    interp = interp1d(data.T[0], data.T[1], kind='linear', axis=0,
                      copy=True, bounds_error=None, fill_value=0.0, assume_sorted=True)
    if f is None:
        f = np.arange(6.0, 48.0, 1.0)
    return interp(f)


coh_control = np.genfromtxt('coherence_control_Popa2013.csv', delimiter=',')
coh_cereboff = np.genfromtxt('coherence_muscimol_Popa2013_Fig3A.csv', delimiter=',')

COH_control = interpolate(coh_control)
COH_cereboff = interpolate(coh_cereboff)

COH = np.vstack([COH_control, COH_cereboff])


plt.figure()
plt.plot(coh_control[:, 0], coh_control[:, 1], 'b--')
plt.plot(coh_cereboff[:, 0], coh_cereboff[:, 1], 'r--')
plt.plot(ftarg, COH.T)

with open('COH.npy', 'wb') as f:
    np.save(f, COH)


# with open('COH.npy', 'rb') as f:
#     COH = np.load(f)

