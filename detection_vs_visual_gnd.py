from ntpath import join
import sys
sys.path.insert(0, "/notebooks/pipenv")
sys.path.insert(0, "/notebooks/nebula3_vlm")
sys.path.insert(0, "/notebooks/nebula3_database")
sys.path.insert(0, "/notebooks/")
sys.path.append("/notebooks/nebula3_playground/vgenome")
sys.path.append("/notebooks/nebula_vg_driver")
sys.path.append("/notebooks/nebula3_vlmtokens_expert/vlmtokens")
import os
import math
import random
import bisect
import pickle
import time
import numpy as np
from nebula3_database.database.arangodb import DatabaseConnector
from nebula3_database.config import NEBULA_CONF

# from movie_db import MOVIE_DB
from pathlib import Path

from nebula3_database.movie_db import MOVIE_DB
# from nebula3_database.playground_db import PLAYGROUND_DB
import tqdm
from PIL import Image
import nebula_vg_driver.visual_genome.local as vg
from nebula3_experts_vg.vg.vg_expert_utils import plot_vg_over_image
import torch
import pandas as pd
import json


class PIPELINE:
    def __init__(self):
        self.config = NEBULA_CONF()
        self.db_host = self.config.get_database_host()
        self.database = self.config.get_playground_name()
        self.gdb = DatabaseConnector()
        self.db = self.gdb.connect_db(self.database)
        self.nre = MOVIE_DB()
        self.nre.change_db("visualgenome")
        self.db = self.nre.db

pipeline = PIPELINE()


# query = 'FOR doc IN objects RETURN doc'
# cursor = pipeline.db.aql.execute(query)
# objects = list(cursor)

# query = 'FOR doc IN images_data RETURN doc'
# cursor = pipeline.db.aql.execute(query)
# images_data = list(cursor)
result_path = "/notebooks/nebula3_playground/images"
res_file = "results_roi_det_vs_vg.csv"
vgenome_images = '/datasets/dataset/vgenome/images/VG_100K'
vgenome_metadata = "/datasets/dataset/vgenome/metadata"
visual_grounding = False 


def get_sc_graph(id):
    return vg.get_scene_graph(id, images=vgenome_metadata,
                    image_data_dir=vgenome_metadata+'/by-id/',
                    synset_file=vgenome_metadata+'/synsets.json')
"""

    vgenome_obj_ontology = list()
    # vgenome_set_obj = set()
    for vg_ob in tqdm.tqdm(vg_objects):
        obj = [x['names'][0] for x in vg_ob['objects']]
        vgenome_obj_ontology.append(obj)


    vgenome_obj_ontology = np.unique(np.concatenate(vgenome_obj_ontology))
    vgenome_obj_ontology = sorted(vgenome_obj_ontology)

"""
# Load ontology
obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

with open(os.path.join(obj_ontology_path, "objects_sorted_all.json"), "r") as f:
    vgenome_obj_ontology = json.load(f)
    vgenome_obj_ontology = np.unique(vgenome_obj_ontology)
    vgenome_obj_ontology = sorted(vgenome_obj_ontology)
# if 1:
#     with open(os.path.join(obj_ontology_path, "objects_sorted_all.json"), "w") as f:
#         vgenome_obj_ontology


from nebula3_vlmtokens_expert.vlmtokens.data.vg_dataset import visual_genome_dataset
from torchvision import transforms

class VisualGenomedatasetRoiLoader(visual_genome_dataset):
    def __init__(self, max_words, indirect_indexing_dictionary=None, indeirect_json_file_path=None, no_roi_extraction = False):
        super(VisualGenomedatasetRoiLoader, self).__init__(max_words=max_words)
        self.transform_ = transforms.Compose([transforms.ToTensor()])
        self.no_roi_extraction = no_roi_extraction
        if indirect_indexing_dictionary is not None and indeirect_json_file is None:
            self.indirect_indexing_dictionary = indirect_indexing_dictionary
            annotation = {}
            skipped_count = 0
            for ix, v in enumerate(self.indirect_indexing_dictionary):
                obj =[x for x in self.ann[0] if x['image_id'] == v][0]
                # print(obj)
            # self.annotation = [self.annotation[int(x)] for x in self.indirect_indexing_dictionary]

                image_id = obj['image_id']
                image_path = os.path.join(self.image_dir,f'{image_id}.{self.image_fmt}')
                if not os.path.exists(image_path):
                    skipped_count += 1
                    continue
                # assume a list of text
                annotation[ix] = {'image': image_path, 'caption':[]}
                #assert isinstance(obj['texts'],list)
                annotation[ix]['objects'] = obj['objects']
                # break        # HK @@ TODO
            annotation = [value for key,value in annotation.items()]
            print('num of images skipped:', skipped_count )
            print('num of images considering:', len(annotation))
# Overrun with the new one
            self.annotation = annotation
            if indeirect_json_file_path is not None:
                print("New image annotation file will be saved to ", indeirect_json_file_path)
                with open(indeirect_json_file_path, "w") as f:
                    json.dump(annotation, f)
        elif indeirect_json_file_path is not None:
            print("New image annotation file will be loaded to ", indeirect_json_file_path)
            with open(indeirect_json_file_path, "r") as f:
                self.annotation =json.load(f)
            
                


        return

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):    
            
        ann = self.annotation[index]
        #print(ann)
        # print(ann['image'])
        image_path = ann['image']        
        image_id = os.path.basename(ann['image'])
        image = Image.open(image_path).convert('RGB')   
        #image = self.transform(image)
        label = []
        croped_images = []
        if self.no_roi_extraction == False:
            #print("OBJECTS ", len(ann['objects']))
            for i, visual_objects in enumerate(ann['objects']):
                h = visual_objects['h']
                w = visual_objects['w']
                y = visual_objects['y']
                x = visual_objects['x']
                x,y,w,h = self.bbox_xywh_to_xyxy((x,y,w,h))
                
                crop_image = image.crop((x,y,w,h))
                width, height = crop_image.size
                if width > 30 and height > 30:
                    #print(width, height)
                    croped_images.append(crop_image)
                    label.append(visual_objects['names'])
        else:
            croped_images.append(image)
            label.append(["full image"])

        processed_frms = [self.transform(frm) for frm in croped_images]
        if not isinstance(processed_frms[0],Image.Image):
            processed_frms = torch.stack(processed_frms)
        return processed_frms, label, image_id

# from torchvision import transforms
# import ruamel.yaml as yaml

from nebula3_videoprocessing.videoprocessing.vlm_factory import VlmFactory

vlm_type = "clip"
vlm = VlmFactory().get_vlm(vlm_type)
results = list()
res_file = "results_bottom_up_" + vlm_type + ".csv"

# Came from  : 
# with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
#     images_data = json.load(f)

# sample_ids = [obj['image_id'] for obj in images_data]
# sample_ids_ipc_paragraph.txt

# with open(os.path.join(vgenome_metadata, "sample_ids_ipc_vgenome_ids.txt"), "r") as f:
#     sample_ids = f.read()
with open(os.path.join(vgenome_metadata, "paragraphs_v1.json"), "r") as f:
    images_data = json.load(f)
sample_ids = np.loadtxt(os.path.join(vgenome_metadata, "sample_ids_ipc_vgenome_ids.txt"))
image_ids_related_to_ipc = [images_data[int(ix)]['image_id'] for ix in sample_ids]

video_dataset = VisualGenomedatasetRoiLoader(max_words=64, indirect_indexing_dictionary=image_ids_related_to_ipc,
                                                indeirect_json_file_path=os.path.join(vgenome_metadata , "vgenome_image_data_related_to_1000_ipc_experiemnt.json"))
# for vg_ob in tqdm.tqdm(sample_ids):
#     cur_image_data = images_data[sample_ids[idx]]
#     print("Object id", cur_image_data['image_id'])

# ontology_imp = SingleOntologyImplementation('vg_objects', 'clip')
if 1: # stats of objects 
    label_stat = list()
    for batch in video_dataset:
        images_batch, label, image_id = batch
        label_stat.append(label)
    label_stat = sorted(label_stat)
    lbl_unique ,count = np.unique(label_stat, return_counts=True)
    ord_idx = np.argsort(count)[::-1]


for batch in video_dataset:
    images_batch, label, image_id = batch
    # print(images_batch)
    # print(label)
    obj_dict = dict()

    for idx, img in enumerate(images_batch):
        bbox = [0, 0, img.shape[1], img.shape[2]] 

        obj_dict = dict()
        for obj in tqdm.tqdm(vgenome_obj_ontology):
            vlm_sim = vlm.compute_similarity(img, text=label[idx][0])
            # vlm_sim = ontology_imp.compute_scores(img)
            obj_dict.update({obj: float(vlm_sim)})
            plot_vg_over_image(bbox=bbox, frame_=img.permute(1, 2, 0), 
                                caption=str(vlm_type) + '_' + label[idx][0] + '_'+str(vlm_sim[0]), 
                                lprob=vlm_sim, path=result_path, create_window=False)
        print("save  dict")
        obj_dict.update({'image_id' :image_id})
        results.append(obj_dict)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_path, res_file), index=False)

df = pd.DataFrame(results)
df.to_csv(os.path.join(result_path, res_file), index=False)

# bottom_up_roi = True

# if bottom_up_roi == True:
#     for vg_ob in tqdm.tqdm(vg_objects):
#         print("Object id",vg_ob['image_id'])



if visual_grounding == True:
    with open(os.path.join(vgenome_metadata, "results-visualgenome.json"), "r") as f:
        vg_objects = json.load(f)

    from nebula3_experts_vg.vg.vg_expert import VisualGroundingVlmImplementation
    vgnd = VisualGroundingVlmImplementation()
    results = list()
    for vg_ob in tqdm.tqdm(vg_objects):
        print("Object id",vg_ob['image_id'])
        fname = os.path.basename(vg_ob['image_url'])
        full_fname = os.path.join(vgenome_images, fname)
        img = Image.open(full_fname)
        img = img.convert('RGB')
        # print(full_fname)
        # sg = get_sc_graph(vg_ob['image_id'])
        obj_dict = dict()
        for obj in tqdm.tqdm(vgenome_obj_ontology):
            bb, lprob = vgnd.compute_similarity(img, obj)
            obj_dict.update({obj: float(lprob)})
        obj_dict.update({'image_id' : vg_ob['image_id']})
        results.append(obj_dict)
    # Intermediate save 
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(result_path, res_file), index=False)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_path, res_file), index=False)


"""

import os
import urllib
temp_file = "vgnoem_tmp_file"
for obj in objects:
    print("Object id",obj['image_id'])
    for img_record in images_data:
        print("obj in image", img_record['image_id'])
        if obj['image_id'] == img_record['image_id']:
            try:
                urllib.request.urlretrieve(img_record['url'], temp_file)
                print("downloaded", img_record['url'])
            except:
                print("Can not download the image ", img_record['url'])
            # filename = wget.download(img_record['url'])
        break
    break



# In[24]:


from visual_genome import api
id = 1
img = api.get_image_data(id)
roi = api.get_region_descriptions_of_image(id)
#sg = api.get_scene_graph_of_image(id)


# In[26]:


np.array(img)


# In[39]:


import json
with open("image_data.json", "r") as f:
    data = json.load(f)


# In[40]:


data[0]


# In[ ]:





# In[38]:


from visual_genome import api
from visual_genome import utils

# import json
# with open("image_data.json", "r") as f:
#     data = json.load(f)
# data = utils.retrieve_data()

# from visual_genome import api as vg
import matplotlib.pyplot as plt
id = 1
image = api.get_image_data(id)

page = 1
next = '/api/v0/images/all?page=' + str(page)
ids = []
data = utils.retrieve_data(next)

regions = utils.parse_region_descriptions(data, image)
print ("The first region descriptions is: %s" % regions[0].phrase)
print ("It is located in a bounding box specified by x:%d, y:%d, width:%d, height:%d" % (regions[0].x, regions[0].y, regions[0].width, regions[0].height))

image = api.get_image_data(id)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
def visualize_regions(image, regions):
    response = requests.get(image.url)
    img = PIL_Image.open(StringIO(response.content))
    plt.imshow(img)
    ax = plt.gca()
    for region in regions:
        ax.add_patch(Rectangle((region.x, region.y),
                               region.width,
                               region.height,
                               fill=False,
                               edgecolor='red',
                               linewidth=3))
        ax.text(region.x, region.y, region.phrase, style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    fig = plt.gcf()
    plt.tick_params(labelbottom='off', labelleft='off')
    plt.show()
visualize_regions(image, regions[:8])


# In[ ]:





# In[ ]:





# In[36]:


data[0]


# In[ ]:





# In[ ]:





# In[ ]:


np.array(img).size()

from visual_genome import api
ids = api.get_all_image_ids()
print (ids[0])
image = api.get_image_data(id=61512)
print (image)


api.get_all_image_ids()
images_data[0]

"""
