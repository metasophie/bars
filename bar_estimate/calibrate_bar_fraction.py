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

FILTER = [150, 200, 356, 444]

N_BINS_z = 5
bins_z = [0, 1, 2, 3, 4, 5]

N_BINS_M = 5
bins_M = [7., 8. ,9., 10., 11., 12.]

N_VOLS = 100
N_RUNS = 100


# different p_feature, binned by z
num_disks = np.zeros((4, N_BINS_z, N_RUNS))
num_barred_disks = np.zeros((4, N_BINS_z, N_RUNS))

for k in range(4):

    result = pd.read_csv(f"bar_estimate/F{FILTER[k]}W_sampling.csv")

    feature_count = result['feature_count'].values
    edgeon_count = result['edgeon_count'].values
    bar_count = result['bar_count'].values

    z50 = result['zfit_50'].values
    M50 = result['logM_50'].values
    q = result['q'].values
    for i in range(len(result)):
        for j in range(N_BINS_z):
            if (z50[i] >= bins_z[j]) & (z50[i] < bins_z[j+1]):
                feature = str2array(feature_count[i])
                edgeon = str2array(edgeon_count[i])
                bar = str2array(bar_count[i])

                # is_disk = (feature >= 0.3*N_VOLS) & (edgeon <= 0.5*feature) & (q[i] >= 0.5)
                is_disk = (feature >= 0.3*N_VOLS) & (feature-edgeon >= 15) & (q[i] >= 0.5)
                num_disks[k,j,:] += is_disk.astype(int)

                is_barred_disk = is_disk & (bar >= 0.5*(feature-edgeon))
                num_barred_disks[k,j,:] += is_barred_disk.astype(int)
            
                break

bar_fraction = num_barred_disks / num_disks
err = np.sqrt(np.sum(bar_fraction*(1-bar_fraction)/num_disks, axis=2)/N_RUNS**2 + np.var(bar_fraction, axis=2))

bin_centers = [(bins_z[i] + bins_z[i+1]) / 2 for i in range(N_BINS_z)]

# plt.plot(bin_centers, np.mean(bar_fraction, axis=1), linestyle='-')
# plt.scatter(bin_centers, np.mean(bar_fraction, axis=1), marker='s')
plt.figure()
for k in range(4):
    plt.errorbar(bin_centers, np.mean(bar_fraction[k,:,:], axis=1), yerr=err[k,:], 
                 fmt='--s', capsize = 5, 
                 label=f'F{FILTER[k]}W')
plt.xticks(bins_z)
plt.xlabel(r"$z$")
plt.ylabel("Bar fraction")
plt.ylim((0, 0.3))
plt.legend(loc='upper right')
plt.savefig(f'bar_estimate/f_bar_z_all_filters.png')


# different p_feature, binned by M
num_disks = np.zeros((4, N_BINS_M, N_RUNS))
num_barred_disks = np.zeros((4, N_BINS_M, N_RUNS))

for k in range(4):

    result = pd.read_csv(f"bar_estimate/F{FILTER[k]}W_sampling.csv")

    feature_count = result['feature_count'].values
    edgeon_count = result['edgeon_count'].values
    bar_count = result['bar_count'].values

    z50 = result['zfit_50'].values
    M50 = result['logM_50'].values
    q = result['q'].values

    for i in range(len(result)):
        for j in range(N_BINS_M):
            if (M50[i] >= bins_M[j]) & (M50[i] < bins_M[j+1]):
                feature = str2array(feature_count[i])
                edgeon = str2array(edgeon_count[i])
                bar = str2array(bar_count[i])

                # is_disk = (feature >= 0.3*N_VOLS) & (edgeon <= 0.5*feature) & (q[i] >= 0.5)
                is_disk = (feature >= 0.3*N_VOLS) & (feature-edgeon >= 15) & (q[i] >= 0.5)
                num_disks[k,j,:] += is_disk.astype(int)

                is_barred_disk = is_disk & (bar >= 0.5*(feature-edgeon))
                num_barred_disks[k,j,:] += is_barred_disk.astype(int)
            
                break

bar_fraction = num_barred_disks / num_disks
err = np.sqrt(np.sum(bar_fraction*(1-bar_fraction)/num_disks, axis=2)/N_RUNS**2 + np.var(bar_fraction, axis=2))

bin_centers = [(bins_M[i] + bins_M[i+1]) / 2 for i in range(N_BINS_M)]

plt.figure()
for k in range(4):
    plt.errorbar(bin_centers, np.mean(bar_fraction[k,:,:], axis=1), yerr=err[k,:], 
                 fmt='--s', capsize = 5, 
                 label=f'F{FILTER[k]}W')
plt.xticks(bins_M)
plt.xlabel(r"$\log{M_*/M_\odot}$")
plt.ylabel("Bar fraction")
plt.ylim((0, 0.3))
plt.legend(loc='upper left')
plt.savefig(f'bar_estimate/f_bar_M_all_filters.png')



N_BINS_mag = 5
bins_mag = [20, 21, 22, 23, 24, 25]

N_BINS_Re = 4
bins_Re = [0., 0.2, 0.4, 0.6, 0.8]


result = pd.read_csv(f"bar_estimate/F200W_sampling.csv")
result = result[(result['zfit_50'] <= 1.) & (result['logM_50'] > 9.) & (result['logM_50'] < 10.)]
# result = result[(result['zfit_50'] <= 1.) & (result['logM_50'] > 10.) & (result['logM_50'] < 11.)]


feature_count = result['feature_count'].values
edgeon_count = result['edgeon_count'].values
bar_count = result['bar_count'].values
id = result['id'].values

cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"

cat = pd.read_csv(os.path.join(cat_dir,cat_name))
mag_F200W = cat['F200W_MAG'].values

Re_F200W = cat['F200W_RE'].values

feature_thresholds = [0.2, 0.3, 0.4]
bar_thresholds = [0.4, 0.5, 0.6]

# z<1, binned by mag_F200W, varying p_bar threshold
num_disks = np.zeros((3, N_BINS_Re, N_RUNS))
num_barred_disks = np.zeros((3, N_BINS_Re, N_RUNS))

for k in range(3):
    for i in range(len(result)):
        for j in range(N_BINS_Re):
            if (Re_F200W[id[i]] >= bins_Re[j]) & (Re_F200W[id[i]] < bins_Re[j+1]):
                feature = str2array(feature_count[i])
                edgeon = str2array(edgeon_count[i])
                bar = str2array(bar_count[i])

                # is_disk = (feature >= 0.3*N_VOLS) & (edgeon <= 0.5*feature) & (q[i] >= 0.5)
                is_disk = (feature >= 0.3*N_VOLS) & (feature-edgeon >= 15) & (q[i] >= 0.5)
                num_disks[k,j,:] += is_disk.astype(int)

                is_barred_disk = is_disk & (bar >= bar_thresholds[k]*(feature-edgeon))
                num_barred_disks[k,j,:] += is_barred_disk.astype(int)
            
                break
print(num_disks, flush=True)
bar_fraction = num_barred_disks / num_disks
err = np.sqrt(np.sum(bar_fraction*(1-bar_fraction)/num_disks, axis=2)/N_RUNS**2 + np.var(bar_fraction, axis=2))

bin_centers = [(bins_Re[i] + bins_Re[i+1]) / 2 for i in range(N_BINS_Re)]

plt.figure()
for k in range(3):
    plt.errorbar(bin_centers, np.mean(bar_fraction[k,:,:], axis=1), yerr=err[k,:], 
                    fmt='--s', capsize = 5, 
                    label=r'$p_\mathrm{bar}\geq%.1f$'%bar_thresholds[k])
plt.xticks(bins_Re)
plt.xlabel("Re F200W/arcsec")
plt.ylabel("Bar fraction")
plt.ylim((0, 1.))
plt.legend(loc='upper right')
plt.savefig(f'bar_estimate/f_bar_F200W_Re_p_bar.png')


# z<1, binned by mag_F200W, varying p_feature threshold
num_disks = np.zeros((3, N_BINS_mag, N_RUNS))
num_barred_disks = np.zeros((3, N_BINS_mag, N_RUNS))

for k in range(3):
    for i in range(len(result)):
        for j in range(N_BINS_mag):
            if (mag_F200W[id[i]] >= bins_mag[j]) & (mag_F200W[id[i]] < bins_mag[j+1]):
                feature = str2array(feature_count[i])
                edgeon = str2array(edgeon_count[i])
                bar = str2array(bar_count[i])

                # is_disk = (feature >= 0.3*N_VOLS) & (edgeon <= 0.5*feature) & (q[i] >= 0.5)
                is_disk = (feature >= feature_thresholds[k]*N_VOLS) & (feature-edgeon >= 15) & (q[i] >= 0.5)
                num_disks[k,j,:] += is_disk.astype(int)

                is_barred_disk = is_disk & (bar >= 0.5*(feature-edgeon))
                num_barred_disks[k,j,:] += is_barred_disk.astype(int)
            
                break

bar_fraction = num_barred_disks / num_disks
err = np.sqrt(np.sum(bar_fraction*(1-bar_fraction)/num_disks, axis=2)/N_RUNS**2 + np.var(bar_fraction, axis=2))

bin_centers = [(bins_mag[i] + bins_mag[i+1]) / 2 for i in range(N_BINS_mag)]

plt.figure()
for k in range(3):
    plt.errorbar(bin_centers, np.mean(bar_fraction[k,:,:], axis=1), yerr=err[k,:], 
                    fmt='--s', capsize = 5, 
                    label=r'$p_\mathrm{feature}\geq%.1f$'%feature_thresholds[k])
plt.xticks(bins_mag)
plt.xlabel("Mag F200W")
plt.ylabel("Bar fraction")
plt.ylim((0, 1.))
plt.legend(loc='upper right')
plt.savefig(f'bar_estimate/f_bar_F200W_mag_p_feature.png')