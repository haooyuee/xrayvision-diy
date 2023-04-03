import collections
import os
import os.path
import pprint
import random
import sys
import tarfile
import warnings
import zipfile

import imageio
import numpy as np
import pandas as pd
import skimage
from typing import Dict
import skimage.transform
from skimage.io import imread
import torch
from torchvision import transforms

def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img

def apply_transforms(sample, transform, seed=None) -> Dict:
    '''
    can remove
    '''
    """Applies transforms to the image and masks.
    The seeds are set so that the transforms that are applied
    to the image are the same that are applied to each mask.
    This way data augmentation will work for segmentation or 
    other tasks which use masks information.
    """

    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        sample["img"] = transform(sample["img"])

        if "pathology_masks" in sample:
            for i in sample["pathology_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["pathology_masks"][i] = transform(sample["pathology_masks"][i])

        if "semantic_masks" in sample:
            for i in sample["semantic_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["semantic_masks"][i] = transform(sample["semantic_masks"][i])

    return sample

'''
need change
'''
class Dataset():
    def __init__(self):
        pass

    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))
    '''
    def __repr__(self):
        if utils.in_notebook():
            pprint.pprint(self.totals())
        return self.string()
    '''
    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view

class NIH_Dataset(Dataset):
    """NIH ChestX-ray8 dataset

    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

    Download resized (224x224) images here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(self,
                 imgpath,
                 csvpath="csvdata/Data_Entry_2017_v2020.csv",
                 bbox_list_path="csvdata/BBox_List_2017.csv",
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True,
                 pathology_masks=False
                 ):
        super(NIH_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()

        self.csv = self.csv.reset_index()

        ####### pathology masks ########
        '''
        can remove not our task
        '''
        # load nih pathology masks
        self.pathology_maskscsv = pd.read_csv(bbox_list_path,
                                              names=["Image Index", "Finding Label", "x", "y", "w", "h", "_1", "_2", "_3"],
                                              skiprows=1)

        # change label name to match
        self.pathology_maskscsv.loc[self.pathology_maskscsv["Finding Label"] == "Infiltrate", "Finding Label"] = "Infiltration"
        self.csv["has_masks"] = self.csv["Image Index"].isin(self.pathology_maskscsv["Image Index"])

        ####### pathology masks ########
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # add consistent csv values

        # offset_day_int
        # self.csv["offset_day_int"] =

        # patientid
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)

        # age
        self.csv['age_years'] = self.csv['Patient Age'] * 1.0

        # sex
        self.csv['sex_male'] = self.csv['Patient Gender'] == 'M'
        self.csv['sex_female'] = self.csv['Patient Gender'] == 'F'

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_mask_dict(self, image_name, this_size):
        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.pathology_maskscsv[self.pathology_maskscsv["Image Index"] == image_name]
        path_mask = {}

        for i in range(len(images_with_masks)):
            row = images_with_masks.iloc[i]

            # Don't add masks for labels we don't have
            if row["Finding Label"] in self.pathologies:
                mask = np.zeros([this_size, this_size])
                xywh = np.asarray([row.x, row.y, row.w, row.h])
                xywh = xywh * scale
                xywh = xywh.astype(int)
                mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

                # Resize so image resizing works
                mask = mask[None, :, :]

                path_mask[self.pathologies.index(row["Finding Label"])] = mask
        return path_mask
