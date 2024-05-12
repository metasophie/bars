import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def str2array(string):

    numbers_str = string.strip('[]').split(',')
    array = np.array([float(num) for num in numbers_str])
    return array

# load catalog
cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"
cat = pd.read_csv(os.path.join(cat_dir,cat_name))
z = cat['zfit_50'].values

# directory for images
image_dir1 = "/scratch/ydong/stamps/demo_F200W"
image_dir2 = "/scratch/ydong/stamps/demo_F200W_added"

# load the finetuned Zoobot predictions 
pred_path = "results/finetune_tree_result/F200W/demo_tree_predictions_F200W_1.csv"
pred = pd.read_csv(pred_path)

id = pred['id_str'].values

alpha_feature_pred = np.array([np.mean(str2array(item)) for item in pred['t0_smooth_or_featured__features_or_disk_pred'].values])
alpha_smooth_pred = np.array([np.mean(str2array(item)) for item in pred['t0_smooth_or_featured__smooth_pred'].values])
alpha_artifact_pred = np.array([np.mean(str2array(item)) for item in pred['t0_smooth_or_featured__star_artifact_or_bad_zoom_pred'].values])
p_feature_pred = alpha_feature_pred/(alpha_feature_pred+alpha_smooth_pred+alpha_artifact_pred)
pred_feature = np.argmax(np.stack((alpha_smooth_pred, alpha_feature_pred, alpha_artifact_pred)), axis=0)


alpha_edgeon_pred = np.array([np.mean(str2array(item)) for item in pred['t2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk_pred'].values])
alpha_else_pred = np.array([np.mean(str2array(item)) for item in pred['t2_could_this_be_a_disk_viewed_edgeon__no_something_else_pred'].values])
pred_edgeon = np.argmax(np.stack((alpha_edgeon_pred, alpha_else_pred)), axis=0)


alpha_strong_pred = np.array([np.mean(str2array(item)) for item in pred['t4_is_there_a_bar__strong_bar_pred'].values])
alpha_weak_pred = np.array([np.mean(str2array(item)) for item in pred['t4_is_there_a_bar__weak_bar_pred'].values])
alpha_none_pred = np.array([np.mean(str2array(item)) for item in pred['t4_is_there_a_bar__no_bar_pred'].values])
p_strong_pred = alpha_strong_pred/(alpha_strong_pred+alpha_weak_pred+alpha_none_pred)
p_weak_pred = alpha_weak_pred/(alpha_strong_pred+alpha_weak_pred+alpha_none_pred)
pred_bar = np.argmax(np.stack((alpha_none_pred, alpha_weak_pred+alpha_strong_pred)), axis=0)


p_bar_pred = p_strong_pred+p_weak_pred


match_path = "bot/match_catalog_F200W.csv"
match_cat = pd.read_csv(match_path)
match_id = match_cat['id_str'].values

count_feature_vol = match_cat['t0_smooth_or_featured__features_or_disk'].values
count_smooth_vol = match_cat['t0_smooth_or_featured__smooth'].values
count_artifact_vol = match_cat['t0_smooth_or_featured__star_artifact_or_bad_zoom'].values
p_feature_vol = count_feature_vol/(count_feature_vol+count_smooth_vol+count_artifact_vol)
true_feature = np.argmax(np.stack((count_smooth_vol, count_feature_vol, count_artifact_vol)), axis=0)

flag_feature = count_feature_vol+count_smooth_vol+count_artifact_vol>25

count_edgeon_vol = match_cat['t2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk'].values
count_else_vol = match_cat['t2_could_this_be_a_disk_viewed_edgeon__no_something_else'].values
true_edgeon = np.argmax(np.stack((count_edgeon_vol, count_else_vol)), axis=0)

flag_edgeon = count_edgeon_vol+count_else_vol>10

count_strong_vol = match_cat['t4_is_there_a_bar__strong_bar'].values
count_weak_vol = match_cat['t4_is_there_a_bar__weak_bar'].values
count_none_vol = match_cat['t4_is_there_a_bar__no_bar'].values
# p_strong_vol = count_strong_vol/(count_strong_vol+count_weak_vol+count_none_vol)
# p_weak_vol = count_weak_vol/(count_strong_vol+count_weak_vol+count_none_vol)
true_bar = np.argmax(np.stack((count_none_vol, count_weak_vol+count_strong_vol)), axis=0)

# p_bar_vol = p_strong_vol+p_weak_vol

flag_bar = count_strong_vol+count_weak_vol+count_none_vol>10

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

common_id, index1, index2 = np.intersect1d(id, match_id[flag_feature], return_indices=True)

cm1 = confusion_matrix(true_feature[flag_feature][index2], pred_feature[index1])
labels_feature = ['Smooth', 'Featured', 'Artifact']
sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax[0], 
            xticklabels=labels_feature, yticklabels=labels_feature)
ax[0].set_title('Smooth or Featured')
ax[0].set_xlabel('Predicted label')
ax[0].set_ylabel('True label')


common_id, index1, index2 = np.intersect1d(id, match_id[flag_edgeon], return_indices=True)

cm2 = confusion_matrix(true_edgeon[flag_edgeon][index2], pred_edgeon[index1])
labels_feature = ['Yes', 'No']
sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", ax=ax[1], 
            xticklabels=labels_feature, yticklabels=labels_feature)
ax[1].set_title('Is Edge-on Disk')
ax[1].set_xlabel('Predicted label')
ax[1].set_ylabel('True label')


common_id, index1, index2 = np.intersect1d(id, match_id[flag_bar], return_indices=True)

cm3 = confusion_matrix(true_bar[flag_bar][index2], pred_bar[index1])
labels_bar = ['None', 'Weak/Strong']
sns.heatmap(cm3, annot=True, fmt="d", cmap="Blues", ax=ax[2], 
            xticklabels=labels_bar, yticklabels=labels_bar)
ax[2].set_title('Has Bar')
ax[2].set_xlabel('Predicted label')
ax[2].set_ylabel('True label')

plt.tight_layout()
plt.savefig('bar_estimate/F200W_test/cm.jpg')