from torchvision.datasets import MNIST, SVHN, SEMEION, USPS
import numpy as np
from PIL import Image
import random
        

class iCOMMNIST():
    def __init__(self,root,
                train_1=True,
                train_2 = 'train',
                transform = None,
                target_transform=None,
                test_transform = None,
                task_type='domain_class',
                local_clients=10, 
                class_fraction = 0.3,
                download=True):
        super(iCOMMNIST,self).__init__()
        self.mnist = MNIST(root, train=train_1, transform=transform, target_transform=target_transform, download=download)
        self.svhn = SVHN(root, split=train_2, transform=transform, target_transform=target_transform, download=download)
        self.semeion = SEMEION(root, transform=transform, target_transform=target_transform, download=download)
        self.usps = USPS(root, train=train_1, transform=transform, target_transform=target_transform, download=download)
        self.task_type = task_type
        self.local_clients = local_clients
        #self.class_fraction = class_fraction
        self.transform = transform
        self.test_transform = test_transform
        #self.target_transform = target_transform

    
    def getTrainData(self, classes, exemplar_set, exemplar_label_set, task_id, client_index):
        datas,labels=[],[]
        if len(exemplar_set)!=0 and len(exemplar_label_set)!=0:
            datas=[exemplar for exemplar in exemplar_set]
            length=len(datas[0])
            labels=[np.full((length), label) for label in exemplar_label_set]
        
        if self.task_type == 'domain_class':
            for label in classes:
                if client_index % 4 == 0:
                    data_mnist = self.mnist.data[np.array(self.mnist.targets)==label]
                    datas.append(np.expand_dims(data_mnist,axis=3).repeat(3,axis=3))
                    labels.append(np.full((data_mnist.shape[0]),label))

                elif client_index % 4 == 1:
                    data_svhn = self.svhn.data[np.array(self.svhn.labels)==label]
                    datas.append(data_svhn.transpose(0, 2, 3, 1))
                    labels.append(np.full((data_svhn.shape[0]),label))

                elif client_index % 4 == 2:
                    sample_index = np.array(self.semeion.labels[0:int(0.85 * len(self.semeion))])==label
                    data_semeion = self.semeion.data[0:int(0.85 * len(self.semeion))][sample_index]
                    datas.append(np.expand_dims(data_semeion,axis=3).repeat(3,axis=3))
                    datas.append(data_semeion)
                    labels.append(np.full((data_semeion.shape[0]),label))

                elif client_index % 4 == 3:
                    data_usps = self.usps.data[np.array(self.usps.targets)==label]
                    datas.append(data_usps)
                    labels.append(np.full((data_usps.shape[0]),label))
                
            self.TrainData, self.TrainLabels = self.concatenate(datas,labels)
        elif self.task_type == 'class_domain':
            for label in classes:
                if task_id % 4 == 0:
                    data_mnist = self.mnist.data[np.array(self.mnist.targets)==label]
                    datas.append(np.expand_dims(data_mnist,axis=1).repeat(3,axis=1))
                    labels.append(np.full((data_mnist.shape[0]),label))

                elif task_id % 4 == 1:
                    data_svhn = self.svhn.data[np.array(self.svhn.labels)==label]
                    datas.append(data_svhn)
                    labels.append(np.full((data_svhn.shape[0]),label))

                elif task_id % 4 == 2:

                    sample_index = np.array(self.semeion.labels[0:int(0.85 * len(self.semeion))])==label
                    data_semeion = self.semeion.data[0:int(0.85 * len(self.semeion))][sample_index]
                    datas.append(data_semeion)
                    labels.append(np.full((data_semeion.shape[0]),label))

                elif task_id % 4 == 3:
                    data_usps = self.usps.data[np.array(self.usps.targets)==label]
                    datas.append(data_usps)
                    labels.append(np.full((data_usps.shape[0]),label))
            self.TrainData, self.TrainLabels = self.concatenate(datas,labels)

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data_mnist = self.mnist.data[np.array(self.mnist.targets)==label]
            datas.append(np.expand_dims(data_mnist,axis=1).repeat(3,axis=1))
            labels.append(np.full((data_mnist.shape[0]),label))

            data_svhn = self.svhn.data[np.array(self.svhn.labels)==label]
            datas.append(data_svhn)
            labels.append(np.full((data_svhn.shape[0]),label))

            data_semeion = self.semeion.data[int(0.85 * len(self.semeion)):][np.array(self.semeion.labels[int(0.85 * len(self.semeion)):])==label]
            datas.append(data_semeion)
            labels.append(np.full((data_semeion.shape[0]),label))

            data_usps = self.usps.data[np.array(self.usps.targets)==label]
            datas.append(data_usps)
            labels.append(np.full((data_usps.shape[0]),label))
            
        self.TestData, self.TestLabels=self.concatenate(datas,labels)  


    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label
    
    def getTrainItem(self,index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

    

        return index,img,target
    
    def getTestItem(self,index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img=self.test_transform(img)
            #img = img.expand(3, img.size(1), img.size(2))

    

        return index, img, target
    
    def __getitem__(self, index):
        if self.TrainData!=[]:
            return self.getTrainItem(index)
        elif self.TestData!=[]:
            return self.getTestItem(index)

    def __len__(self):
        if self.TrainData!=[]:
            return len(self.TrainData)
        elif self.TestData!=[]:
            return len(self.TestData)
    



            
                

        
