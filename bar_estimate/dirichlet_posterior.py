import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import betabinom
from scipy.special import gamma, beta

def str2array(string):

    numbers_str = string.strip('[]').split(',')
    array = np.array([float(num) for num in numbers_str])
    return array


def generalized_beta_binomial_pmf(x, n:int, a, b):

    return gamma(n+1)/gamma(x+1)/gamma(n-x+1)*beta(x+a,n-x+b)/beta(a,b)

FILTER = 150

# load catalog
cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"
cat = pd.read_csv(os.path.join(cat_dir,cat_name))
z50 = cat['zfit_50'].values
z16 = cat['zfit_16'].values
z84 = cat['zfit_84'].values
q = cat[f'F{FILTER}W_Q'].values
q_err = cat[f'F{FILTER}W_Q_ERR'].values


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
    # p_feature_pred = alpha_feature_pred/(alpha_feature_pred+alpha_smooth_pred+alpha_artifact_pred)

    alpha_edgeon_pred.append(pred['t2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk_pred'].values)
    alpha_else_pred.append(pred['t2_could_this_be_a_disk_viewed_edgeon__no_something_else_pred'].values)

    alpha_strong_pred.append(pred['t4_is_there_a_bar__strong_bar_pred'].values)
    alpha_weak_pred.append(pred['t4_is_there_a_bar__weak_bar_pred'].values)
    alpha_none_pred.append(pred['t4_is_there_a_bar__no_bar_pred'].values)
    # p_strong_pred = alpha_strong_pred/(alpha_strong_pred+alpha_weak_pred+alpha_none_pred)
    # p_weak_pred = alpha_weak_pred/(alpha_strong_pred+alpha_weak_pred+alpha_none_pred)

    # p_bar_pred = p_strong_pred+p_weak_pred


match_path = f"bot/match_catalog_F{FILTER}W.csv"
match_cat = pd.read_csv(match_path)
match_id = match_cat['id_str'].values
image_loc = match_cat['file_loc'].values

count_feature_vol = match_cat['t0_smooth_or_featured__features_or_disk'].values
count_smooth_vol = match_cat['t0_smooth_or_featured__smooth'].values
count_artifact_vol = match_cat['t0_smooth_or_featured__star_artifact_or_bad_zoom'].values
total_feature_question_vol = count_feature_vol+count_smooth_vol+count_artifact_vol
p_feature_vol = count_feature_vol/(count_feature_vol+count_smooth_vol+count_artifact_vol)

count_edgeon_vol = match_cat['t2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk'].values
count_else_vol = match_cat['t2_could_this_be_a_disk_viewed_edgeon__no_something_else'].values
total_disk_question_vol = count_edgeon_vol+count_else_vol
p_edgeon_vol = count_edgeon_vol/(count_edgeon_vol+count_else_vol)

count_strong_vol = match_cat['t4_is_there_a_bar__strong_bar'].values
count_weak_vol = match_cat['t4_is_there_a_bar__weak_bar'].values
count_none_vol = match_cat['t4_is_there_a_bar__no_bar'].values
total_bar_question_vol = count_strong_vol+count_weak_vol+count_none_vol
p_strong_vol = count_strong_vol/(count_strong_vol+count_weak_vol+count_none_vol)
p_weak_vol = count_weak_vol/(count_strong_vol+count_weak_vol+count_none_vol)

p_bar_vol = p_strong_vol+p_weak_vol


common_id, index1, index2 = np.intersect1d(id, match_id, return_indices=True)
# print(index1,index2)

color_list = ['orange', 'green', 'violet']

for i in np.random.randint(len(common_id), size=20):
# for i in np.where((alpha_strong_pred + alpha_weak_pred) > 2*alpha_none_pred)[0]:
    
    plt.figure(figsize=(20,5))
    plt.subplot(1, 4, 1)
    plt.imshow(mpimg.imread(image_loc[index2[i]]), cmap='gray')
    plt.title(f'F{FILTER}W image')
    plt.text(10, 20, r"$z=%.2f^{+%.2f}_{-%.2f}$"%(z50[common_id[i]], z50[common_id[i]]-z16[common_id[i]], z84[common_id[i]]-z50[common_id[i]]),
             fontsize=12, ha='left', color='white')
    plt.text(10, 40, r"$q=%.2f\pm%.2f$"%(q[common_id[i]], q_err[common_id[i]]),
             fontsize=12, ha='left', color='white')

    n1 = int(total_feature_question_vol[index2[i]])
    x1 = np.linspace(0, n1, 100*n1)

    pdf_feature = np.zeros((15, 100*n1))

    plt.subplot(1, 4, 2)
    plt.xlim((0,n1))
    plt.ylim(bottom=0)
    for j in range(3):
        a_feature = str2array(alpha_feature_pred[j][i])
        b_feature = str2array(alpha_smooth_pred[j][i]) + str2array(alpha_artifact_pred[j][i])
        for k in range(5):
            pdf_feature[j*5+k,:] = generalized_beta_binomial_pmf(x1, n1, a_feature[k], b_feature[k])
            plt.plot(x1, pdf_feature[j*5+k,:], color=color_list[j], alpha=0.3)
    plt.plot(x1, np.mean(pdf_feature, axis=0), color='blue', alpha=1)
    plt.axvline(p_feature_vol[index2[i]]*n1, color='black', linestyle='dashed')
    plt.title(r'Feature votes')

    n2 = int(total_disk_question_vol[index2[i]])
    x2 = np.linspace(0, n2, 100*n2)

    pdf_disk = np.zeros((15, 100*n2))

    plt.subplot(1, 4, 3)
    plt.xlim((0,n2))
    plt.ylim(bottom=0)
    for j in range(3):
        a_disk = str2array(alpha_edgeon_pred[j][i])
        b_disk = str2array(alpha_else_pred[j][i])
        for k in range(5):
            pdf_disk[j*5+k,:] = generalized_beta_binomial_pmf(x2, n2, a_disk[k], b_disk[k])
            plt.plot(x2, pdf_disk[j*5+k,:], color=color_list[j], alpha=0.3)
    plt.plot(x2, np.mean(pdf_disk, axis=0), color='blue', alpha=1)
    plt.axvline(p_edgeon_vol[index2[i]]*n2, color='black', linestyle='dashed')
    plt.title(r'Edge-on votes')

    n3 = int(total_bar_question_vol[index2[i]])
    x3 = np.linspace(0, n3, 100*n3)

    pdf_bar = np.zeros((15, 100*n3))

    plt.subplot(1, 4, 4)
    plt.xlim((0,n3))
    plt.ylim(bottom=0)
    for j in range(3):
        a_bar = str2array(alpha_strong_pred[j][i]) + str2array(alpha_weak_pred[j][i])
        b_bar = str2array(alpha_none_pred[j][i])
        for k in range(5):
            pdf_bar[j*5+k,:] = generalized_beta_binomial_pmf(x3, n3, a_bar[k], b_bar[k])
            plt.plot(x3, pdf_bar[j*5+k,:], color=color_list[j], alpha=0.3)
    plt.plot(x3, np.mean(pdf_bar, axis=0), color='blue', alpha=1)
    plt.axvline(p_bar_vol[index2[i]]*n3, color='black', linestyle='dashed')
    plt.title(r'Bar votes')

    plt.tight_layout()
    plt.savefig(f'bar_estimate/F{FILTER}W_pred/F{FILTER}W_dirichlet_id{common_id[i]}.png')

