import sys
sys.path.insert(0, "/notebooks/pipenv")
sys.path.insert(0, "/notebooks/nebula3_vlm")
sys.path.insert(0, "/notebooks/nebula3_database")
sys.path.insert(0, "/notebooks/")
sys.path.append("/notebooks/nebula3_playground/vgenome")
sys.path.append("/notebooks/nebula_vg_driver")
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
import cv2
from pathlib import Path

from nebula3_database.movie_db import MOVIE_DB
# from nebula3_database.playground_db import PLAYGROUND_DB
import tqdm
from PIL import Image
from nebula3_experts_vg.vg.vg_expert import VisualGroundingVlmImplementation
import nebula_vg_driver.visual_genome.local as vg
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
result_path = "/notebooks/nebula3_playground"
res_file = "results_roi_det_vs_vg.csv"
vgenome_images = '/datasets/dataset/vgenome/images/VG_100K'
VG_DATA = "/datasets/dataset/vgenome/metadata"
with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
    vg_objects = json.load(f)

def get_sc_graph(id):
    return vg.get_scene_graph(id, images=VG_DATA,
                    image_data_dir=VG_DATA+'/by-id/',
                    synset_file=VG_DATA+'/synsets.json')
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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


sg


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


np.array(img).size()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


from visual_genome import api
ids = api.get_all_image_ids()
print (ids[0])
image = api.get_image_data(id=61512)
print (image)


# In[8]:


api.get_all_image_ids()


# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


images_data[0]


# In[ ]:




"""
