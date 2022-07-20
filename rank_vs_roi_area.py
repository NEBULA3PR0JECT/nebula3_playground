import pandas as pd 
import os
import matplotlib.pyplot as plt
import numpy as np
csv_path = '/datasets/dataset/vgenome/results'
result_path = "/notebooks/nebula3_playground/images"

# csv_file = 'results_roi_det_vs_vg_min_h_60_min_w_60.csv' #'results_bottom_up_clip.csv'
topk = 30
csv_file = 'res_stat_rank_vs_area_' + str(topk) + '.csv'
df = pd.read_csv(os.path.join(csv_path, csv_file), index_col=False)

df.rank_norm_to_topk[df.rank_norm_to_topk==-1] = 0
df.rank_nl[df.rank_nl.isna()] = 0

plt.figure()
plt.scatter(np.log10(df.area), df.rank_norm_to_topk, s=1)

if 0:
    plt.scatter(np.log10(df.area), df.rank_nl, s=1)

# plt.scatter(df.area, df.rank_norm_to_topk, s=1)
# plt.scatter(np.log10(df.area), np.log10(df.rank_norm_to_topk+ 1e-7), s=1)


# plt.title('Scatter plot Topk : ()', topk)
plt.title('Scatter plot Topk : {}'.format(topk))
plt.xlabel('log10 [Bbox area]')
plt.ylabel('Ranked GT [1/rank]')
plt.grid()
plt.ylim([0.0, 1.05])

# Create names on the x axis
plt.savefig(os.path.join(result_path, 'object_hist_1000_ipc_bottom_up_len_' + str(topk) + '.png'))
