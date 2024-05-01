import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import rv_discrete, betabinom
import numpy as np

def sample_posterior(n:int, posterior, n_samples:int=1):
    assert n == np.size(posterior)-1
    # assert np.sum(posterior) == 1.
    if np.sum(posterior) != 1:
        for i in range(n+1):
            if posterior[i] > np.sum(posterior)-1:
                posterior[i] += 1-np.sum(posterior)
                break

    rv = rv_discrete(values=(range(n+1), posterior))
    sample = rv.rvs(size=n_samples)
    return sample
    # if np.isscalar(sample):
    #     return int(sample)
    # else:
    #     return sample.astype(int)


def str2array(string):

    numbers_str = string.strip('[]').split(',')
    array = np.array([float(num) for num in numbers_str])
    return array

def array2str(array):
    str_list = [str(num) for num in array]
    string = '[' + ','.join(str_list) + ']'
    return string


FILTER = 200
N_RUNS = 100
N_VOLS = 100

sampling_result = pd.DataFrame(columns=['id', 'zfit_50', 'zfit_16', 'zfit_84', 'q', 'q_err', 
                                        'logM_50', 'logM_16', 'logM_84', 'feature_count', 'edgeon_count', 'bar_count'])

# image_dir = f'/scratch/ydong/stamps/demo_F{FILTER}W'
# file_loc = [os.path.join(image_dir,path) for path in os.listdir(image_dir)]
# ids = np.array([int(re.findall(r'\d+',path)[1]) for path in os.listdir(image_dir)])

cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"
cat = pd.read_csv(os.path.join(cat_dir,cat_name))

z50 = cat['zfit_50'].values
z16 = cat['zfit_16'].values
z84 = cat['zfit_84'].values

q = cat['F200W_Q'].values
q_err = cat['F200W_Q_ERR'].values

m50 = cat['logM_50'].values
m16 = cat['logM_16'].values
m84 = cat['logM_84'].values

alpha_feature_pred = []
alpha_smooth_pred = []
alpha_artifact_pred = []

alpha_edgeon_pred = []
alpha_else_pred = []

alpha_strong_pred = []
alpha_weak_pred = []
alpha_none_pred = []

for i in range(3):
    # load the finetuned Zoobot predictions 
    pred_path = f"bar_estimate/F{FILTER}W_pred/full_cat_predictions_F{FILTER}W_{i}.csv"
    pred = pd.read_csv(pred_path)

    id = pred['id_str'].values

    alpha_feature_pred.append(pred['t0_smooth_or_featured__features_or_disk_pred'].values)
    alpha_smooth_pred.append(pred['t0_smooth_or_featured__smooth_pred'].values)
    alpha_artifact_pred.append(pred['t0_smooth_or_featured__star_artifact_or_bad_zoom_pred'].values)

    alpha_edgeon_pred.append(pred['t2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk_pred'].values)
    alpha_else_pred.append(pred['t2_could_this_be_a_disk_viewed_edgeon__no_something_else_pred'].values)

    alpha_strong_pred.append(pred['t4_is_there_a_bar__strong_bar_pred'].values)
    alpha_weak_pred.append(pred['t4_is_there_a_bar__weak_bar_pred'].values)
    alpha_none_pred.append(pred['t4_is_there_a_bar__no_bar_pred'].values)

start = time.time()

for i in range(len(id)):
    feature_count = np.zeros(N_RUNS, dtype=int)
    edgeon_count = np.zeros(N_RUNS, dtype=int)
    bar_count = np.zeros(N_RUNS, dtype=int)

    pmf_feature = np.zeros((15, N_VOLS+1))
    for j in range(3):
        a_feature = str2array(alpha_feature_pred[j][i])
        b_feature = str2array(alpha_smooth_pred[j][i]) + str2array(alpha_artifact_pred[j][i])
        for l in range(5):
            pmf_feature[j*5+l,:] = betabinom.pmf(range(N_VOLS+1), N_VOLS, a_feature[l], b_feature[l])
    mean_pmf_feature = np.mean(pmf_feature, axis=0)
    # mean_pmf_feature = np.round(mean_pmf_feature, decimals=5)
    # mean_pmf_feature = mean_pmf_feature / np.sum(mean_pmf_feature)
    # print(np.sum(mean_pmf_feature))
    feature_count = sample_posterior(N_VOLS, mean_pmf_feature, n_samples=N_RUNS)

    for k in range(N_RUNS):

        N_FEATURE = feature_count[k]
        pmf_edgeon = np.zeros((15, N_FEATURE+1))
        for j in range(3):
            a_disk = str2array(alpha_edgeon_pred[j][i])
            b_disk = str2array(alpha_else_pred[j][i])
            for l in range(5):
                pmf_edgeon[j*5+l,:] = betabinom.pmf(range(N_FEATURE+1), N_FEATURE, a_disk[l], b_disk[l])
        mean_pmf_edgeon = np.mean(pmf_edgeon, axis=0)
        edgeon_count[k] = sample_posterior(N_FEATURE, mean_pmf_edgeon)

        N_FACEON = feature_count[k] - edgeon_count[k]
        pmf_bar = np.zeros((15, N_FACEON+1))
        for j in range(3):
            a_bar = str2array(alpha_strong_pred[j][i]) + str2array(alpha_weak_pred[j][i])
            b_bar = str2array(alpha_none_pred[j][i])
            for l in range(5):
                pmf_bar[j*5+l,:] = betabinom.pmf(range(N_FACEON+1), N_FACEON, a_bar[l], b_bar[l])
        mean_pmf_bar = np.mean(pmf_bar, axis=0)
        bar_count[k] = sample_posterior(N_FACEON, mean_pmf_bar)
        
    item = pd.DataFrame({'id': id[i], 'zfit_50': z50[id[i]], 'zfit_16': z16[id[i]], 'zfit_84': z84[id[i]], 
                         'q': q[id[i]], 'q_err': q_err[id[i]], 'logM_50': m50[id[i]], 'logM_16': m16[id[i]], 'logM_84': m84[id[i]], 
                         'feature_count': array2str(feature_count), 'edgeon_count': array2str(edgeon_count), 'bar_count': array2str(bar_count)}, 
                         index=[i])
    sampling_result = pd.concat([sampling_result, item])

    if i%100 == 0:
        end = time.time()
        print(i, end-start, flush=True)

sampling_result.to_csv(f'bar_estimate/F{FILTER}W_sampling.csv')

