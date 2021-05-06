import numpy as np 
from PIL import Image
from torchvision import transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader

from utils.gaussian_blur import GaussianBlur

class Data:
    def __init__(self, dataset):
        self.dataset = dataset
        self.load_datasets()

        # setup dataTransform
        color_jitter = transforms.ColorJitter(0.4,0.4,0.4,0.1)
        self.train_transforms = transforms.Compose([transforms.RandomResizedCrop(size = 224,scale=(0.5, 1.0)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p = 0.7),
                                            transforms.RandomGrayscale(p  = 0.2),
                                            GaussianBlur(3),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                                            ])
        self.test_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                         
        ])
        self.test_cifar10_transforms = transforms.Compose([
                                            transforms.Resize((224, 224)),  
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                         
        ])
    
    def load_datasets(self):
        raise NotImplementedError

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        train_dataset = MyTrainDataset(self.X_train, self.Y_train, self.train_transforms)

        if(self.dataset == 'cifar10'):
            val_dataset = MyTestDataset(self.X_val, self.Y_val, self.test_cifar10_transforms, self.dataset)
            test_dataset = MyTestDataset(self.X_test, self.Y_test, self.test_cifar10_transforms, self.dataset)
            database_dataset = MyTestDataset(self.X_database, self.Y_database, self.test_cifar10_transforms, self.dataset)
        else:
            val_dataset = MyTestDataset(self.X_val, self.Y_val, self.test_transforms, self.dataset)
            test_dataset = MyTestDataset(self.X_test, self.Y_test, self.test_transforms, self.dataset)
            database_dataset = MyTestDataset(self.X_database, self.Y_database, self.test_transforms, self.dataset)

        # DataLoader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                shuffle=shuffle_train,
                                                num_workers=num_workers)

        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers)

        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers) if get_test else None

        database_loader = DataLoader(dataset=database_dataset, batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers)
        
        return train_loader, val_loader, test_loader, database_loader

class LabeledData(Data):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)
    
    def load_datasets(self):
        if(self.dataset == 'cifar10'):
            self.topK = 1000
            self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.X_database, self.Y_database = get_cifar()
        else:
            raise NotImplementedError("Please use the right dataset!")

class MyTrainDataset(Dataset):
    def __init__(self,data,labels, transform):
        self.data = data
        self.labels = labels
        self.transform  = transform
    def __getitem__(self, index):
        pilImg = Image.fromarray(self.data[index])
        imgi = self.transform(pilImg)
        imgj = self.transform(pilImg)
        return (imgi, imgj, self.labels[index])
    
    def __len__(self):
        return len(self.data)

class MyTestDataset(Dataset):
    def __init__(self,data,labels, transform,dataset):
        self.data = data
        self.labels = labels
        self.transform  = transform
        self.dataset = dataset
    def __getitem__(self, index):
        if self.dataset == 'cifar10':
            pilImg = Image.fromarray(self.data[index])
            return (self.transform(pilImg),self.labels[index])
        else:
            return (self.transform(self.data[index]),self.labels[index])
        
    def __len__(self):
        return len(self.data)

def get_cifar():
    # Dataset
    train_dataset = dsets.CIFAR10(root='./data/cifar10/',
                                train=True,
                                download=True)

    test_dataset = dsets.CIFAR10(root='./data/cifar10/',
                                train=False
                                )

    database_dataset = dsets.CIFAR10(root='./data/cifar10/',
                                    train=True
                                    )


    # train with 5000 images
    X = train_dataset.data
    L = np.array(train_dataset.targets)

    first = True
    for label in range(10):
        index = np.where(L == label)[0]
        N = index.shape[0]
        prem = np.random.permutation(N)
        index = index[prem]
        
        data = X[index[0:500]]
        labels = L[index[0: 500]]
        if first:
            Y_train = labels
            X_train = data
        else:
            Y_train = np.concatenate((Y_train, labels))
            X_train = np.concatenate((X_train, data))
        first = False

    Y_train = np.eye(10)[Y_train]

    
    idxs = list(range(len(test_dataset.data)))
    np.random.shuffle(idxs)
    test_data = np.array(test_dataset.data)
    test_tragets = np.array(test_dataset.targets)

    X_val = test_data[idxs[:5000]]
    Y_val = np.eye(10)[test_tragets[idxs[:5000]]]

    X_test = test_data[idxs[5000:]]
    Y_test = np.eye(10)[test_tragets[idxs[5000:]]]


    X_database = database_dataset.data 
    Y_database = np.eye(10)[database_dataset.targets]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database

