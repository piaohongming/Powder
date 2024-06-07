from torchvision.datasets import MNIST, SVHN, SEMEION, USP, CelebA
import numpy as np
from PIL import Image
import random
import os

class iCELEBA(CelebA):
    def __init__(self,root,
                 train='train',
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True,
                 human_task=300,
                 human_client=30):
        super(iCELEBA,self).__init__(root,
                                       split=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)
        self.target_test_transform = target_test_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.human_task = human_task
        self.human_client = human_client

    def getTestData(self, task_id, client_index):
        self.TestData = self.filename
        self.TestLabels = self.attr

    def getTrainData(self, task_id, client_index):
        upper_bound = task_id * self.human_task + (client_index + 1) * self.human_client
        lower_bound = task_id * self.human_task + client_index * self.human_client
        #datas,labels=[],[]
        self.TrainData = self.filename[np.array(self.identity) < upper_bound and np.array(self.identity) >= upper_bound]
        self.TrainLabels = self.attr[np.array(self.identity) < upper_bound and np.array(self.identity) >= upper_bound]



    def getTrainItem(self,index):
        img, target = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.TrainData[index])), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

    

        return index,img,target

    def getTestItem(self, index):
        img, target = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.TestData[index])), self.TestLabels[index]
        if self.test_transform:
            img=self.test_transform(img)
        
        return index, img, target