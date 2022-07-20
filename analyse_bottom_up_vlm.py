import pandas as pd
import os
import sys
sys.path.insert(0, "/notebooks/pipenv")
sys.path.insert(0, "/notebooks/nebula3_database")
sys.path.insert(0,"/notebooks/nebula3_experiments")
sys.path.insert(0,"/notebooks/nebula_vg_driver")
sys.path.insert(0,"/notebooks/nebula_vg_driver/visual_genome")
sys.path.insert(0, "/notebooks/")

from nebula3_experiments import vg_eval
import tqdm
import numpy as np
import torch
# import json
from nebula3_experiments.vg_eval import VGEvaluation#, get_sc_graph, spice_get_triplets, tuples_from_sg
from ast import literal_eval

# pip install spacy
# pip install spacy-wordnet
# python -m spacy download en
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_lg


evaluator = VGEvaluation()
filter_area = True
min_area = 60*60
if filter_area:
    print("Filter min area", min_area)

csv_path = '/datasets/dataset/vgenome/results'
# csv_file = 'results_roi_det_vs_vg_min_h_60_min_w_60.csv' #'results_bottom_up_clip.csv'
csv_file = 'results_roi_det_vs_vg_min_h_60_min_w_60_clip.csv'
df = pd.read_csv(os.path.join(csv_path, csv_file))

for top_k in range(30, 50, 5):
    print("top_k", top_k)
    res_stat_rank_vs_area = list()
    # recall = evaluator.recall_triplets_mean(gt_objects, ipc_objects)
    image_ids = np.unique(df['image_id'])
    df['area'] = df['roi_coord'].apply(lambda x: literal_eval(x)[2]*literal_eval(x)[3])
    # if 1:
    df_vlm_scores_per_object = df.loc[:, ~df.columns.isin(['image_id', 'ground_truth', 'roi_coord', 'area'])]
    df_vlm_scores_per_object_with_gt = df.loc[:, ~df.columns.isin(['image_id', 'roi_coord'])]
    # else:
    #     df_vlm_scores_per_object = df.loc[:, ~df.columns.isin(['image_id', 'ground_truth', 'roi_coord'])]
    #     df_vlm_scores_per_object_with_gt = df.loc[:, ~df.columns.isin(['image_id', 'roi_coord'])]

    df_vlm_scores_per_object_with_gt['gt_score'] = df_vlm_scores_per_object_with_gt.apply(lambda x: x['ground_truth'] if x['ground_truth'] not in x else x[x['ground_truth']] , axis=1)
    # coord = df['roi_coord']
    # coord = [literal_eval(x) for x in coord]
    obj_list = list(df_vlm_scores_per_object.keys())
    recall_per_image = dict()

    for img_idx, ids in enumerate(tqdm.tqdm(image_ids)):
        if filter_area:
            df_ = df_vlm_scores_per_object[df['image_id'] == ids][df['area']> min_area]
            df_vlm_scores_per_object_with_gt_id = df_vlm_scores_per_object_with_gt[df['image_id'] == ids][df['area']> min_area]
            gt = df[df['image_id'] == ids]['ground_truth'][df['area']> min_area].to_list()
        else:
            df_ = df_vlm_scores_per_object[df['image_id'] == ids]
            df_vlm_scores_per_object_with_gt_id = df_vlm_scores_per_object_with_gt[df['image_id'] == ids]
            gt = df[df['image_id'] == ids]['ground_truth'].to_list()
        roi_per_image_vlm_score = torch.from_numpy(df_.to_numpy().astype('float32'))
        v_top_k, i_topk = torch.topk(roi_per_image_vlm_score, k=top_k, dim=1) # worst cast if only one ROI/BB than it has to prsent the same top-k
        # objects_mat = np.tile(np.array(obj_list),(len(df_), 1))
        pred_vlm = [np.array(obj_list)[j] for j in np.array(i_topk)]
        # round rubun raster to fond the top_k out of N BB
        n_gt = len(gt)

        # 
# Stat per roi
        if 1:
            df_vlm_scores_per_object_with_gt_id = df_vlm_scores_per_object_with_gt_id[['gt_score', 'area']]
            sorted, indices = torch.sort(roi_per_image_vlm_score)
            v_top_k_np = np.array(sorted)
            for ix, (ix_df, row) in enumerate(df_vlm_scores_per_object_with_gt_id.iterrows()):    
                stat = dict()    
                cond = row.gt_score > v_top_k_np[ix, :][::-1] # top down
                # print(cond)
                if np.any(np.where(cond)[0]):
                    # print(np.where(row.gt_score > v_top_k_np[ix, :])[0])
                    rank = 1 - (np.where(cond))[0][0]/v_top_k_np.shape[1] #rank = 1/(np.where(cond))[0][0]
                    rank_nl = 1/(np.where(cond))[0][0]
                    res_stat_rank_vs_area.append({'area': row.area, 'rank_norm_to_topk' :rank, 'rank_nl': rank_nl})
                else:
                    res_stat_rank_vs_area.append({'area': row.area, 'rank_norm_to_topk' :int(-1)})

        if n_gt >=top_k and 0:
            print("More than N BB/ROI", n_gt)
            top_out_of_bb_list = [x[0] for x in pred_vlm]
            top_out_of_bb_list = np.unique(top_out_of_bb_list)
            top_out_of_bb_list = top_out_of_bb_list[:top_k]

            recall_i = list()
            for ix in range(min(n_gt, top_k)):
                recall_i.append(evaluator.smanager.compare_triplet(gt[ix], top_out_of_bb_list[ix]))
            recall = np.array(recall_i).mean()
        else:
            top = 0
            top_out_of_bb_list = list()

            while len(top_out_of_bb_list)<top_k:
                next_unique_top_k_out_of_bb = np.unique([t[top] for t in pred_vlm])
                top_out_of_bb_list.extend(next_unique_top_k_out_of_bb)
                top += 1

# Trunc according to top-K
            top_out_of_bb_list = top_out_of_bb_list[:top_k]
            gt = [tuple([x]) for x in gt]
            top_out_of_bb_list = [tuple([x]) for x in top_out_of_bb_list]
            recall = evaluator.recall_triplets_mean(src=gt, dst=top_out_of_bb_list)

        if len(top_out_of_bb_list) < top_k:
            print("From some reason could not find top k", len(top_out_of_bb_list))


        # recall_many_2_many = evaluator.recall_triplets_mean(src=gt, dst=top_out_of_bb_list) # not considering which ROi GT
        print(recall)
        recall_per_image.update({ids :recall})
        # df_result = pd.DataFrame.from_dict(recall_per_image, orient='index')
        df_result = pd.DataFrame(recall_per_image.items(), columns=['image', 'recall_mean'])
        if img_idx%10 == 0:
            if filter_area:
                df_result.to_csv(os.path.join(csv_path, 'recall_botoom_up_top_filter_area_' + str(top_k) +'.csv'), index=False)
                df_results_area_rank = pd.DataFrame(res_stat_rank_vs_area)
                df_results_area_rank.to_csv(os.path.join(csv_path, 'res_stat_rank_vs_area_' + str(top_k) + '.csv'), index=False)

            else:
                df_result.to_csv(os.path.join(csv_path, 'recall_botoom_up_top_' + str(top_k) +'.csv'))
                df_results_area_rank = pd.DataFrame(res_stat_rank_vs_area)
                df_results_area_rank.to_csv(os.path.join(csv_path, 'res_stat_rank_vs_area_' + str(top_k) + '.csv'), index=False)

    print("topk {} Mean (bert-based) total recall of ground truth triplets in ipc triplets is: {:3f}".format(top_k, np.mean(df_result['recall_mean'])))


"""
for i in h[0]:
    
for ix, gt_i in enumerate(min(gt, top_k)):
    print(ix)


v_top_k_np = np.array(v_top_k)    
for ix, (ix_df, row) in enumerate(df_vlm_scores_per_object_with_gt.iterrows()):    
    print(np.where(v_top_k_np[ix, :] < row.gt_score))
    break

v_top_k_np = np.array(v_top_k)    
for ix, (ix_df, row) in enumerate(df_vlm_scores_per_object_with_gt.iterrows()):    
    print(np.where(row.gt_score > v_top_k_np[ix, :])[0])


v_top_k_np = np.array(v_top_k)
res_stat = list()
for ix, (ix_df, row) in enumerate(df_vlm_scores_per_object_with_gt.iterrows()):    
    stat = dict()    
    cond = row.gt_score > v_top_k_np[ix, :]
    print(cond)
    if np.any(np.where(cond)[0]):
        print(np.where(row.gt_score > v_top_k_np[ix, :])[0])
        rank = 1/(np.where(cond))[0][0]
        stat.update({row.area :rank})
    else:
        stat.update({row.area :-1})
    res_stat.append(stat)

import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(os.path.join('/notebooks/nebula3_playground/images/blip_itc', 'recall_botoom_up_top_filter_area_30.csv'), index_col=False)
plt.figure()
plt.hist(df.recall_mean, bins=100)
plt.xlabel('recall')
plt.ylabel('occurences')
plt.grid()
plt.title('Histogram BLIP-ITC of recall per image top30 (mu {:3f}, std {:3f}) : '.format(np.mean(df.recall_mean), np.std(df.recall_mean)))
plt.savefig("recall_hist_top30.png")

nroi_per_image = list()
for img_idx, ids in enumerate(tqdm.tqdm(image_ids)):
    df_ = df_vlm_scores_per_object[df['image_id'] == ids][df['area']> min_area]
    nroi_per_image.append(len(df_))
"""