import collections
import os
import os.path
import random
import sys

import numpy as np
import pandas as pd

from typing import Dict
from skimage.io import imread
import torch


class NIH_Dataset():

    def __init__(self,
                 imgpath,
                 csvpath="csvdata/Data_Entry_2017_v2020.csv",
                 bbox_list_path="csvdata/BBox_List_2017.csv",
                 views=None,
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 rseed=0,
                 pathology_masks=False
                 ):
        super(NIH_Dataset, self).__init__()

        self.imgpath = imgpath
        self.csvpath = csvpath
        self.bbox_list_path = bbox_list_path
        self.views = views
        self.transform = transform
        self.data_aug = data_aug
        self.nrows = nrows
        self.rseed = rseed
        self.pathology_masks = pathology_masks

        self.labels = []
        self.pathologies = sorted(["Atelectasis", "Consolidation", "Infiltration",
                                   "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                                   "Effusion", "Pneumonia", "Pleural_Thickening",
                                   "Cardiomegaly", "Nodule", "Mass", "Hernia"])

        np.random.seed(self.rseed)

        self.check_paths()
        self.csv = pd.read_csv(self.csvpath, nrows=self.nrows)

        self.csv["view"] = self.csv['View Position']
        self.preprocess()

    def check_paths(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def preprocess(self):
        self.csv = self.csv.reset_index()

        if self.pathology_masks:
            # load nih pathology masks
            self.pathology_maskscsv = pd.read_csv(self.bbox_list_path,
                                                  names=["Image Index", "Finding Label", "x", "y", "w", "h", "_1", "_2",
                                                         "_3"],
                                                  skiprows=1)

            # change label name to match
            self.pathology_maskscsv.loc[
                self.pathology_maskscsv["Finding Label"] == "Infiltrate", "Finding Label"] = "Infiltration"
            self.csv["has_masks"] = self.csv["Image Index"].isin(self.pathology_maskscsv["Image Index"])

        # Get our classes.

        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # patientid
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)
        self.csv['age_years'] = self.csv['Patient Age'] * 1.0

        self.csv['sex_male'] = self.csv['Patient Gender'] == 'M'
        self.csv['sex_female'] = self.csv['Patient Gender'] == 'F'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image at idx
        sample = {"idx": idx, "lab": self.labels[idx]}
        imgid = self.csv['Image Index'].iloc[idx]
        img = imread(os.path.join(self.imgpath, imgid))

        # Image Normalization
        if img.max() > 255:
            raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), 255))

        img = (2 * (img.astype(np.float32) / 255) - 1.) * 1024

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        sample["img"] = img[None, :, :]

        # Image Transform
        if self.transform is not None:
            sample["img"] = self.transform(sample["img"])

        # Data Augmentation
        if self.data_aug is not None:
            sample["img"] = self.data_aug(sample["img"])

        return sample

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)