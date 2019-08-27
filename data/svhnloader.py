from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from utils import download_url, check_integrity, TransformTwice, RandomTranslateWithReflect
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False, labeled=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        if labeled:
            ind = [i for i in range(len(self.labels)) if int(self.labels[i]) in [0, 1, 2, 3, 4]]
        else:
            ind = [i for i in range(len(self.labels)) if int(self.labels[i]) in [5, 6, 7, 8, 9]]
        
        self.data = self.data[ind]
        self.labels= self.labels[ind]
        #  print(self.data.shape)
        #  print(self.labels.shape)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def SVHNLoader(root, batch_size, split='train', num_workers=2, labeled = True, aug=None, shuffle=True):
    if aug == None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])  
    elif aug == 'once':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])  
    elif aug =='twice':
        transform = TransformTwice(transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]))

    dataset = SVHN(root=root, split=split, labeled=labeled, transform=transform)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

 
def show_batch(inp, title=None):
    import matplotlib.pyplot as plt
    """Show batch"""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()
    raw_input()

if __name__ == "__main__":
    #  import torch
    import torchvision
    from torchvision import datasets, transforms
    #  from collections import Counter
    image_dataset_root = 'data_shallow14/datasets/SVHN'
    dfm = transforms.Compose([
            transforms.RandomAffine(degrees = (-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.2), shear = (-10, 10)),
                        ])
    tsf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
                   ])
    #  dataset = SVHN(image_dataset_root, split='train', transform=tsf, labeled = True, download=True)
    #  dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle= True)
    dataloader_train = SVHNLoader(root=image_dataset_root, batch_size=8, split='train', cls_per_batch=None, num_workers=2, labeled = True, shuffle=True)
    img, target,idx = next(iter(dataloader_train))
    print('target', target)
    print(len(dataloader_train.dataset))
    out = torchvision.utils.make_grid(img)
    show_batch(out, title=target)

