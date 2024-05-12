# Class mapping:
#   HA4K_120 -> 0
#   HB4K_120 -> 1
#   HN4K_120 -> 2
#   MA4K_120 -> 3
#   MB4K_120 -> 4
#   MN4K_120 -> 5
# -*- coding: utf-8 -*-
# TASK 1: Read the DiveFace database and obtain the embeddings of 50 face images (1 image per subject) from 
# the 6 demographic groups (50*6=300 embeddings in total).

import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten
from keras_vggface.vggface import VGGFace
import keras.utils as image

import os
import numpy as np
import pickle
from PIL import Image
import random
from tqdm import tqdm

from labels import class_mapping

def create_model():
    #Import the ResNet-50 model trained with VGG2 database
    my_model = 'resnet50'
    resnet = VGGFace(model = my_model)
    #resnet.summary()  

    #Select the lat leayer as feature embedding  
    last_layer = resnet.get_layer('avg_pool').output
    feature_layer = Flatten(name='flatten')(last_layer)
    model_vgg=Model(resnet.input, feature_layer)
    return model_vgg

if __name__ == '__main__':
    # initialize the model from which we extract the features
    model_vgg = create_model()
    path_to_dataset = '../data/4K_120'
    path_to_save = '../data/embeddings/'
    limit = 100000
    begin = 550
    
    dataset_info = {}
    ds_embeddings = []
    ds_labels = []
    ds_eth_label = []
    ds_gen_label = []
    ds_ids = []
    # for each racegroup in the dataset
    for race_group in os.listdir(path_to_dataset):
        print(f"Processing group {race_group}")
        ids_race_group = os.path.join(path_to_dataset, race_group)
        label = class_mapping(race_group)
        # for each id that belongs to a race group
        counter = 0
        for ids in tqdm(os.listdir(ids_race_group)[begin:begin+limit]):
            # get all the pics from a given id
            pics_ids = os.path.join(ids_race_group, ids)
            name_pics_id = os.listdir(pics_ids)
            n_pics = len(name_pics_id)
            # select a random pic from all of them
            r_idx = np.random.randint(low=0, high=n_pics)
            pic_id = name_pics_id[r_idx]
            pic_id_path = os.path.join(pics_ids, pic_id)
            # open the pic in PIL/numpy format and forward into the model
            pic_frame = Image.open(pic_id_path)
            pic_frame = pic_frame.resize((224, 224), resample=Image.BILINEAR)
            pic_frame = np.array(pic_frame)
            pic_frame = np.expand_dims(pic_frame, axis=0)
            embedding = model_vgg.predict(pic_frame, verbose=0)
            # append the labels
            ds_labels.append([label])
            ds_eth_label.append([label%3])
            ds_gen_label.append([0 if label<3 else 1])
            ds_embeddings.append(embedding)
            ds_ids.append(ids)
            # update counter 
            counter += 1
        
    # shuffle the data
    combined_lists = list(zip(ds_embeddings, ds_labels, ds_eth_label, ds_gen_label, ds_ids))
    random.shuffle(combined_lists)
    ds_embeddings, ds_labels, ds_eth_label, ds_gen_label, ds_ids = zip(*combined_lists)   
    
    # store the values on an array
    dataset_info['embeddings'] = np.concatenate(ds_embeddings, axis=0)
    dataset_info['labels'] = np.concatenate(ds_labels, axis=0)
    dataset_info['eth_labels'] = np.concatenate(ds_eth_label, axis=0)
    dataset_info['gen_labels'] = np.concatenate(ds_gen_label, axis=0)
    dataset_info['ids'] = ds_ids
    
    print(f"Extracted {dataset_info['embeddings'].shape} embeddings.")
    
    path_to_save = os.path.join(path_to_save, f"limit_{limit}")
    if not os.path.exists(os.path.join(path_to_save, f"limit_{limit}")):
        os.makedirs(path_to_save)
        
    with open(os.path.join(path_to_save, "embeddings.pkl"), 'wb') as f:
        pickle.dump(dataset_info, f)
            
            
            
    
        
