# load script
import my_float as myfloat
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats


gradients = np.load(os.path.join(
    'Datasets', 'seismic_exploration_DLR', 'gradients.npy'))
models = np.load(os.path.join(
    'Datasets', 'seismic_exploration_DLR', 'models.npy'))
# dimension [x coord., z coord., receiver no.]

# exemplary image of gradient for receiver 5
plt.figure()
plt.imshow(gradients[:, :, 5].T, aspect='auto')
# exemplary image of velocity model for receiver 5
plt.figure()
plt.imshow(models[:, :, 5].T, aspect='auto')
plt.show()


# Remarks:
# -I need just a p(s) of all floats
# -But note: Here we have definitely more structure than just floats

# import myfunctions as mf
# models.flatten().shape
# gradients.flatten().shape
floatx = myfloat.float_toolbox('float16')

bint, _ = floatx.float2bitint(gradients.flatten())
b_count1 = np.bincount(bint)
p_s = np.zeros(2 ** floatx.N_bits)
p_s[0:b_count1.shape[0]] = b_count1
p_s = p_s / np.sum(p_s)

plt.figure()
plt.plot(floatx.x_poss, p_s, 'r-o', label='p(s)')
plt.ylim((0, 0.04))
plt.xlim((-2.5, 2.5))
plt.figure()
plt.hist(gradients.flatten(), bins=1000)
plt.ylim((0, 10000))
plt.xlim((-1, 1))


if floatx.N_bits < 16:
    data = models.flatten() - np.mean(models.flatten())
    # data = models.flatten() - stats.mode(models.flatten())[0]
else:
    data = models.flatten()
bint, _ = floatx.float2bitint(data)
b_count2 = np.bincount(bint)
p_s2 = np.zeros(2 ** floatx.N_bits)
p_s2[0:b_count2.shape[0]] = b_count2
p_s2 = p_s2 / np.sum(p_s2)

plt.figure()
plt.plot(floatx.x_poss, p_s2, 'r-o', label='p(s)')
# plt.ylim((0, 0.04))
# plt.xlim((-2.5, 2.5))
plt.figure()
plt.hist(models.flatten(), bins=1000)
# plt.ylim((0, 50000))
# plt.xlim((1980, 2200))


# Combination of data
data_comb = np.concatenate((gradients.flatten(), data))
bint, _ = floatx.float2bitint(data_comb)
b_count3 = np.bincount(bint)
p_s3 = np.zeros(2 ** floatx.N_bits)
p_s3[0:b_count3.shape[0]] = b_count3
p_s3 = p_s3 / np.sum(p_s3)

plt.figure()
plt.plot(floatx.x_poss, p_s3, 'r-o', label='p(s)')
# plt.ylim((0, 0.04))
# plt.xlim((-30, 30))
plt.xlim((-100, 2500))
plt.figure()
plt.hist(data_comb, bins=1000)
