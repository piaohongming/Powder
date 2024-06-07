from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
import torch
import torch.utils.data as data
from dataloaders.utils import download_url, check_integrity
import random
import torchvision.datasets as datasets
import yaml
from torchvision.datasets.vision import VisionDataset

class iDataset(data.Dataset):
    
    def __init__(self, root,
                train=True, transform=None,
                download_flag=False,
                seed=-1, validation=False, domain=0):

        # process rest of args
        self.domain = domain
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        #self.tasks = tasks
        self.download_flag = download_flag

        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap labels to match task order
        '''
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1
        '''

        # targets as numpy.array
        self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)

        # if validation
        if self.validation:
            
            # shuffle
            state = np.random.get_state()
            np.random.seed(self.seed)
            randomize = np.random.permutation(len(self.targets))
            self.data = self.data[randomize]
            self.targets = self.targets[randomize]
            np.random.set_state(state)

            # sample
            n_data = len(self.targets)
            if self.train:
                self.data = self.data[:int(0.8*n_data)]
                self.targets = self.targets[:int(0.8*n_data)]
            else:
                self.data = self.data[int(0.8*n_data):]
                self.targets = self.targets[int(0.8*n_data):]

            # train set
            if self.train:
                self.data = self.data[:int(0.8*n_data)]
                self.targets = self.targets[:int(0.8*n_data)]
                '''
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))
                '''

            # val set
            else:
                '''
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))
                '''

        # else
        else:
            '''
            self.archive = []
            domain_i = 0
            for task in self.tasks:
                if True:
                    locs = np.isin(self.targets, task).nonzero()[0]
                    self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))
            '''

        if self.train:
            self.coreset = (np.zeros(0, dtype=self.data.dtype), np.zeros(0, dtype=self.targets.dtype))

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        
        if self.TrainData!=[]:
            img, target = self.TrainData[index], self.TrainLabels[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            return index, img, target
        elif self.TestData!=[]:
            img, target = self.TestData[index], self.TestLabels[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            return index, img, target

    
    def concatenate(self,datas,labels):
        if len(datas) > 0:
            con_data=datas[0]
            con_label=labels[0]
            for i in range(1,len(datas)):
                con_data=np.concatenate((con_data,datas[i]),axis=0)
                con_label=np.concatenate((con_label,labels[i]),axis=0)
        else:
            con_data = np.array([])
            con_label = np.array([])
        return con_data,con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels=self.concatenate(datas,labels)

    def getTestData_hard(self, classes, classes_real):
        datas,labels=[],[]
        for label in classes_real:
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), classes[classes_real.index(label)]))
        self.TestData, self.TestLabels=self.concatenate(datas,labels)
    
    def getDistillTrainData(self, exemplar_dict, classes):
        datas, labels = [],[]
        for c in classes:
            datas.append(exemplar_dict[c])
            length = len(exemplar_dict[c])
            labels.append(np.full((length), c))
        self.TrainData, self.TrainLabels = self.concatenate(datas,labels)    
    
    def getDistillTrainDataLabel(self, labels):
        self.TrainLabels = labels
    
    def getTrainData(self, classes, exemplar_set, exemplar_label_set, client_index, classes_real=None, classes_proportion=None, class_distribution_client_di=None, exe_class=None):
        datas,labels=[],[]
        if len(exemplar_set)!=0 and len(exemplar_label_set)!=0:
            datas=[exemplar for exemplar in exemplar_set]
            length=len(datas[0])
            labels=[np.full((length), label) for label in exemplar_label_set]
            
            datas_preserve = []
            labels_preserve = []
            for c in exe_class:
                if c in exemplar_label_set:
                    datas_preserve.append(datas[exemplar_label_set.index(c)])
                    labels_preserve.append(labels[exemplar_label_set.index(c)])
            datas = datas_preserve
            labels = labels_preserve
        if class_distribution_client_di is None:
            if classes_real is None:
                for label in classes:
                    
                    if self.domain >= 0:
                        data=self.data[np.array(self.targets)==label]
                        data = data[int(classes_proportion[0]*len(data)): int(classes_proportion[1]*len(data))]
                    else:
                        domain_name = list(self.data_by_domain.keys())[client_index]
                        data=np.array(self.data_by_domain[domain_name])[np.array(self.targets_by_domain[domain_name])==label]
                        data = data[int(classes_proportion[0]*len(data)): int(classes_proportion[1]*len(data))]
                    datas.append(data)
                    labels.append(np.full((data.shape[0]),label))
            else:
                for label in classes_real:
                    
                    if self.domain >= 0:
                        data=self.data[np.array(self.targets)==label]
                        data = data[int(classes_proportion[0]*len(data)): int(classes_proportion[1]*len(data))]
                    else:
                        domain_name = list(self.data_by_domain.keys())[client_index]
                        data=np.array(self.data_by_domain[domain_name])[np.array(self.targets_by_domain[domain_name])==label]
                        data = data[int(classes_proportion[0]*len(data)): int(classes_proportion[1]*len(data))]
                    
                    datas.append(data)
                    labels.append(np.full((data.shape[0]),classes[classes_real.index(label)]))
        else:
            if classes_real is None:
                for label in classes:
                    
                    if self.domain >= 0:
                        data=self.data[np.array(self.targets)==label]
                        data = data[random.sample(list(range(len(data))), int(len(data)*class_distribution_client_di[classes.index(label)])+1)]
                    else:
                        domain_name = list(self.data_by_domain.keys())[client_index]
                        data=np.array(self.data_by_domain[domain_name])[np.array(self.targets_by_domain[domain_name])==label]
                        data = data[random.sample(list(range(len(data))), int(len(data)*class_distribution_client_di[classes.index(label)])+1)]
                    datas.append(data)
                    labels.append(np.full((data.shape[0]),label))
            else:
                for label in classes_real:
                    
                    if self.domain >= 0:
                        data=self.data[np.array(self.targets)==label]
                        data = data[random.sample(list(range(len(data))), int(len(data)*class_distribution_client_di[classes_real.index(label)])+1)]
                    else:
                        domain_name = list(self.data_by_domain.keys())[client_index]
                        data=np.array(self.data_by_domain[domain_name])[np.array(self.targets_by_domain[domain_name])==label]
                        data = data[random.sample(list(range(len(data))), int(len(data)*class_distribution_client_di[classes_real.index(label)])+1)]
                    datas.append(data)
                    labels.append(np.full((data.shape[0]),classes[classes_real.index(label)]))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)

    def getTrainImbalance(self, classes, exemplar_set, exemplar_label_set, client_index):
        number_imbalance = []
        for label in range(200):
            if label in exemplar_label_set:
                number_imbalance.append(len(exemplar_set[0]))
            elif label in classes:
                if self.domain >= 0:
                    data=self.data[np.array(self.targets)==label]
                else:
                    domain_name = list(self.data_by_domain.keys())[client_index]
                    data=np.array(self.data_by_domain[domain_name])[np.array(self.targets_by_domain[domain_name])==label]
                
                number_imbalance.append(len(data))
            else:
                number_imbalance.append(0)
            
        number_imbalance = [i / sum(number_imbalance) for i in number_imbalance]
        return number_imbalance

    def getSampleData(self, classes, exemplar_set, exemplar_label_set, group):
        datas,labels=[],[]
        if len(exemplar_set)!=0 and len(exemplar_label_set)!=0:
            datas=[exemplar for exemplar in exemplar_set]
            length=len(datas[0])
            labels=[np.full((length), label) for label in exemplar_label_set]

        if group == 0:
            for label in classes:
                data=self.data[np.array(self.targets)==label]
                datas.append(data)
                labels.append(np.full((data.shape[0]),label))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)


    def load(self):
        pass

    def get_image_class(self,label,client_index, classes_proportion=None):
        if self.domain >= 0:
            data = self.data[np.array(self.targets)==label]
            data = data[int(classes_proportion[0]*len(data)): int(classes_proportion[1]*len(data))]
            return data
        else:
            domain_name = self.data_by_domain.keys()[client_index]
            return self.data_by_domain[domain_name][np.array(self.targets_by_domain[domain_name])==label]
        
    def get_image_class_dict(self,label,client_index, classes_proportion=None, exemplar_set=None):
        if self.domain >= 0:
            data = self.data[np.array(self.targets)==label]
            data = data[int(classes_proportion[0]*len(data)): int(classes_proportion[1]*len(data))]
            datas = [i for i in data]
            if exemplar_set is not None:
                datas = datas + exemplar_set
            return datas
        else:
            domain_name = self.data_by_domain.keys()[client_index]
            return self.data_by_domain[domain_name][np.array(self.targets_by_domain[domain_name])==label]


    def __len__(self):
        if len(self.TrainData) != 0:
            return len(self.TrainData)
        elif len(self.TestData) != 0:
            return len(self.TestData)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class iIMAGENET_R(iDataset):
    
    base_folder = 'imagenet-r'
    im_size=224
    nch=3
    def load(self):

        # load splits from config file
        if self.train or self.validation:
            data_config = yaml.load(open('dataloaders/splits/imagenet-r_train_2.yaml', 'r'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open('dataloaders/splits/imagenet-r_test_2.yaml', 'r'), Loader=yaml.Loader)
        self.data = data_config['data']
        self.targets = data_config['targets']

    def __getitem__(self, index, simple = False):
        if len(self.TrainData) != 0:
            img_path, target = self.TrainData[index], self.TrainLabels[index]
            img = jpg_image_to_array(img_path)

            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            
            
            return index, img, target
        elif len(self.TestData) != 0:
            img_path, target = self.TestData[index], self.TestLabels[index]
            img = jpg_image_to_array(img_path)

            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            return index, img, target
    
    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
    
    def _load_meta(self):
        with open('/home/piaohongming/FCL/Baselines/src/data/imagenet-r/class2label.log') as f:
            dict = {}
            self.class_to_idx = {}
            content = f.read()
            for line in content.split('\n'):
                line_list = line.split(' ')
                dict[line_list[0]] = line_list[1]
            for i in range(len(self.data)):
                class_idx = self.targets[i]
                class_file_name = self.data[i].split('/')[2]
                class_name = dict[class_file_name]
                if class_name in self.class_to_idx.keys():
                    continue
                else:
                    self.class_to_idx[class_name] = class_idx
    
    def _count_domain_class(self):
        print(len(self.data))
        self.class_with_all_domain = {}
        for i in range(len(self.data)):
            class_idx = self.targets[i]
            domain_name = self.data[i].split('/')[3].split('_')[0]
            if class_idx in self.class_with_all_domain.keys():
                if domain_name in self.class_with_all_domain[class_idx]:
                    continue
                else:
                    self.class_with_all_domain[class_idx].append(domain_name)
            else:
                self.class_with_all_domain[class_idx] = []
                self.class_with_all_domain[class_idx].append(domain_name)
        count = 0
        for class_idx in self.class_with_all_domain.keys():
            if len(self.class_with_all_domain[class_idx]) >= 5:
                count = count + 1
                print('class id: {} number of domain: {}'.format(class_idx, len(self.class_with_all_domain[class_idx])))
        print(count)


class iDOMAIN_NET(iIMAGENET_R):
    base_folder = 'DomainNet'
    im_size=224
    nch=3
    def load(self):
        
        # load splits from config file
        if self.train or self.validation:
            data_config = yaml.load(open('./dataloaders/splits/domainnet_train_2.yaml', 'r'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open('./dataloaders/splits/domainnet_test_2.yaml', 'r'), Loader=yaml.Loader)
        self.data = data_config['data']
        self.targets = data_config['targets']
        if self.domain < 0:
            self.data_by_domain = {}
            self.targets_by_domain = {}
            for i in range(len(self.data)):
                domain_name = self.data[i].split('/')[2]
                if domain_name in self.data_by_domain.keys():
                    self.data_by_domain[domain_name].append(self.data[i])
                    self.targets_by_domain[domain_name].append(self.targets[i])
                else:
                    self.data_by_domain[domain_name] = []
                    self.targets_by_domain[domain_name] = []
                    self.data_by_domain[domain_name].append(self.data[i])
                    self.targets_by_domain[domain_name].append(self.targets[i])


    def _load_meta(self):
        self.class_to_idx = {}
        for i in range(len(self.data)):
            class_idx = self.targets[i]
            class_name = self.data[i].split('/')[3]
            if class_name in self.class_to_idx.keys():
                continue
            else:
                self.class_to_idx[class_name] = class_idx

    def _count_domain_class(self):
        self.class_with_all_domain = {}
        for i in range(len(self.data)):
            class_idx = self.targets[i]
            domain_name = self.data[i].split('/')[2]
            if class_idx in self.class_with_all_domain.keys():
                if domain_name in self.class_with_all_domain[class_idx]:
                    continue
                else:
                    self.class_with_all_domain[class_idx].append(domain_name)
            else:
                self.class_with_all_domain[class_idx] = []
                self.class_with_all_domain[class_idx].append(domain_name)
      
        count = 0
        for class_idx in self.class_with_all_domain.keys():
            if len(self.class_with_all_domain[class_idx]) == 5:
                count = count + 1
        print('class_domain: {}'.format(count))
        

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:      
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
    return im_arr