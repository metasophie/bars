import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np

def str2array(string):

    numbers_str = string.strip('[]').split(',')
    array = np.array([int(num) for num in numbers_str])
    return array

FILTER = 200

N_BINS_z = 5
bins_z = [0, 1, 2, 3, 4, 5]

N_BINS_M = 5
bins_M = [7., 8. ,9., 10., 11., 12.]

N_VOLS = 100
N_RUNS = 100

result = pd.read_csv(f"bar_estimate/F{200}W_sampling.csv")

feature_count = result['feature_count'].values
edgeon_count = result['edgeon_count'].values
bar_count = result['bar_count'].values

z50 = result['zfit_50'].values
M50 = result['logM_50'].values

num_disks = np.zeros((N_BINS_z, N_RUNS))
num_barred_disks = np.zeros((N_BINS_z, N_RUNS))

for i in range(len(result)):
    for j in range(N_BINS_z):
        if (z50[i] >= bins_z[j]) & (z50[i] < bins_z[j+1]):
            feature = str2array(feature_count[i])
            edgeon = str2array(edgeon_count[i])
            bar = str2array(bar_count[i])

            is_disk = (feature >= 0.3*N_VOLS) & (edgeon <= 0.5*feature)
            num_disks[j,:] += is_disk.astype(int)

            is_barred_disk = is_disk & (bar >= 0.5*(feature-edgeon))
            num_barred_disks[j,:] += is_barred_disk.astype(int)
        
            break

bar_fraction = num_barred_disks / num_disks
print(np.sum(bar_fraction*(1-bar_fraction)/num_disks, axis=1)/N_RUNS, np.var(bar_fraction, axis=1))
err = np.sqrt(np.sum(bar_fraction*(1-bar_fraction)/num_disks, axis=1)/N_RUNS + np.var(bar_fraction, axis=1))

bin_centers = [(bins_z[i] + bins_z[i+1]) / 2 for i in range(N_BINS_z)]

# plt.plot(bin_centers, np.mean(bar_fraction, axis=1), linestyle='-')
# plt.scatter(bin_centers, np.mean(bar_fraction, axis=1), marker='s')
plt.figure()
plt.errorbar(bin_centers, np.mean(bar_fraction, axis=1), yerr=np.std(bar_fraction, axis=1), fmt='--s')
plt.xticks(bins_z)
plt.xlabel(r"$z$")
plt.ylabel("Bar fraction")
plt.ylim((0,1.))
plt.savefig('bar_estimate/f_bar_z.png')



num_disks = np.zeros((N_BINS_M, N_RUNS))
num_barred_disks = np.zeros((N_BINS_M, N_RUNS))

for i in range(len(result)):
    for j in range(N_BINS_M):
        if (M50[i] >= bins_M[j]) & (M50[i] < bins_M[j+1]):
            feature = str2array(feature_count[i])
            edgeon = str2array(edgeon_count[i])
            bar = str2array(bar_count[i])

            is_disk = (feature >= 0.3*N_VOLS) & (edgeon <= 0.5*feature)
            num_disks[j,:] += is_disk.astype(int)

            is_barred_disk = is_disk & (bar >= 0.5*(feature-edgeon))
            num_barred_disks[j,:] += is_barred_disk.astype(int)
        
            break

bar_fraction = num_barred_disks / num_disks
print(np.sum(bar_fraction*(1-bar_fraction)/num_disks, axis=1)/N_RUNS, np.var(bar_fraction, axis=1))
err = np.sqrt(np.sum(bar_fraction*(1-bar_fraction)/num_disks, axis=1)/N_RUNS + np.var(bar_fraction, axis=1))

bin_centers = [(bins_M[i] + bins_M[i+1]) / 2 for i in range(N_BINS_M)]

# plt.plot(bin_centers, np.mean(bar_fraction, axis=1), linestyle='-')
# plt.scatter(bin_centers, np.mean(bar_fraction, axis=1), marker='s')
plt.figure()
plt.errorbar(bin_centers, np.mean(bar_fraction, axis=1), yerr=err, fmt='--s')
plt.xticks(bins_M)
plt.xlabel(r"$\log{M_*/M_\odot}$")
plt.ylabel("Bar fraction")
plt.ylim((0,1.))
plt.savefig('bar_estimate/f_bar_M.png')