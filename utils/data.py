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
    train_dataset = dsets.CIFAR10(
        root='./data/cifar10/',
        train=True,
        download=True
    )
    test_dataset = dsets.CIFAR10(
        root='./data/cifar10/',
        train=False
    )

    train_size = 5000
    val_size = 500
    test_size = 500

    # train with 5000 images
    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            val_index = index[:val_size]
            test_index = index[val_size:val_size + test_size]
            train_index = index[val_size + test_size: val_size + test_size + train_size]
            database_index = index[val_size + test_size + train_size:]
        else:
            val_index = np.concatenate((val_index, index[:val_size]))
            test_index = np.concatenate((test_index, index[val_size:val_size + test_size]))
            train_index = np.concatenate((train_index, index[val_size + test_size: val_size + test_size + train_size]))
            database_index = np.concatenate((database_index, index[val_size + test_size + train_size:]))
        first = False

    database_index = np.concatenate((train_index, database_index))  # DeepHash cifar10-2

    X_train = X[train_index]
    Y_train = np.eye(10)[L[train_index]]
    X_val = X[val_index]
    Y_val = np.eye(10)[L[val_index]]
    X_test = X[test_index]
    Y_test = np.eye(10)[L[test_index]]
    X_database = X[database_index]
    Y_database = np.eye(10)[L[database_index]]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database
