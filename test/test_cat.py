'''
Y. Dong, Sept 6
test opening the CEERS catalog
'''

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"


cat = pd.read_csv(os.path.join(cat_dir,cat_name))

class_dir = "/scratch/ydong/classifications"
class_name = "jwst-ceers-v0-5-aggregated-class-singlechoicequestionsonly.csv"

cla = pd.read_csv(os.path.join(class_dir,class_name))

col_names = ['RA_1','DEC_1']
cols = cat.columns

id = pd.read_csv("bar_estimate/F200W_sampling.csv")['id'].values

M = cat['logM_50'].values[id]
z = cat['zfit_50'].values[id]

g = sns.jointplot(x=z, y=M, kind='kde', joint_kws={'levels':8})
g.set_axis_labels(r'$z$', r'$\log{M_*/M_\odot}$', fontsize=12)
g.ax_joint.set_xlim([0, 6])
g.ax_joint.set_ylim([6, 12])
plt.tight_layout()
plt.savefig('joint.jpg')

# for col in cols:
#     print(col)

# print(cla[['pixrad','radius_select','flux_rad_0p50','which_nircam','nircam_id']].values)
# print(np.unique(cla['which_nircam'].values))

# for i in range(100,200):
#     print(cat[['ID','zfit_50','zfit_16','zfit_84','logM_50','logM_16','logM_84','logMt_50','logMt_16','logMt_84']].values[i])
