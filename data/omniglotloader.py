from __future__ import print_function
from PIL import Image
from os.path import join
import os
import torch.utils.data as data
from utils import download_url, check_integrity, list_dir, list_files
import torch
import torchvision
from torchvision import transforms
from sampler import RandSubClassSampler

class Omniglot(data.Dataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
    folder = 'omniglot-py'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    def __init__(self, root, background=True,
                 transform=None, target_transform=None,
                 download=False, deform=None):
        self.root = join(os.path.expanduser(root), self.folder)
        self.background = background
        self.transform = transform
        self.deform = deform
        self.target_transform = target_transform

        if download:
            self.download()

        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                for a in self._alphabets], [])
        self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
                                  for idx, character in enumerate(self._characters)]
        self._flat_character_images = sum(self._character_images, [])

    def __len__(self):
        return len(self._flat_character_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self._flat_character_images[index]
        image_path = join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        image_d = image
        if self.deform is not None:
            image_d = self.deform(image_d) 
        if self.transform:
            image = self.transform(image)
            image_d = self.transform(image_d)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, image_d, character_class, index

    def _check_integrity(self):
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
            return False
        return True

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        filename = self._get_target_folder()
        zip_filename = filename + '.zip'
        url = self.download_url_prefix + '/' + zip_filename
        download_url(url, self.root, zip_filename, self.zips_md5[filename])
        print('Extracting downloaded file: ' + join(self.root, zip_filename))
        with zipfile.ZipFile(join(self.root, zip_filename), 'r') as zip_file:
            zip_file.extractall(self.root)

    def _get_target_folder(self):
        if self.background == 'images_background_train' or self.background == 'images_background_val':
            return self.background 
        return 'images_background' if self.background else 'images_evaluation'

def Omniglot_loader(batch_size, num_workers=2, root='../data'):
    binary_flip = transforms.Lambda(lambda x: 1 - x)
    normalize = transforms.Normalize((0.086,), (0.235,))
    train_dataset = Omniglot(
        root=root, download=True, background=True,
        transform=transforms.Compose(
           [transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize]
        ))
    train_length = len(train_dataset)
    train_imgid2cid = [train_dataset[i][2] for i in range(train_length)]  # train_dataset[i] returns (img, cid)
    # Randomly select 20 characters from 964. By default setting (batch_size=100), each character has 5 images in a mini-batch.
    train_sampler = RandSubClassSampler(
        inds=range(train_length),
        labels=train_imgid2cid,
        cls_per_batch=20,
        batch_size=batch_size,
        num_batch=train_length//batch_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, sampler=train_sampler)
    train_loader.num_classes = 964

    test_dataset = Omniglot(
        root=root, download=True, background=False,
        transform=transforms.Compose(
          [transforms.Resize(32),
           transforms.ToTensor(),
           binary_flip,
           normalize]
        ))
    eval_length = len(test_dataset)
    eval_imgid2cid = [test_dataset[i][2] for i in range(eval_length)]
    eval_sampler = RandSubClassSampler(
        inds=range(eval_length),
        labels=eval_imgid2cid,
        cls_per_batch=20,
        batch_size=batch_size,
        num_batch=eval_length // batch_size)
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, sampler=eval_sampler)
    eval_loader.num_classes = 659

    return train_loader, eval_loader

def Omniglot_bg_loader(batch_size, num_workers=2, train_cls_per_batch=20, test_cls_per_batch=20, root='../data'):
    binary_flip = transforms.Lambda(lambda x: 1 - x)
    normalize = transforms.Normalize((0.086,), (0.235,))
    train_dataset = Omniglot(
        root=root, download=False, background='images_background_train',
        transform=transforms.Compose(
           [transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize]
        ))

    if train_cls_per_batch is not None:
        train_length = len(train_dataset)
        train_imgid2cid = [train_dataset[i][2] for i in range(train_length)]  
        train_sampler = RandSubClassSampler(
            inds=range(train_length),
            labels=train_imgid2cid,
            cls_per_batch=train_cls_per_batch,
            batch_size=batch_size,
            num_batch=train_length//batch_size)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers, sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    train_loader.num_classes = 964 - 169

    test_dataset = Omniglot(
        root=root, download=False, background='images_background_val',
        transform=transforms.Compose(
          [transforms.Resize(32),
           transforms.ToTensor(),
           binary_flip,
           normalize]
        ))
    if test_cls_per_batch is not None:
        eval_length = len(test_dataset)
        eval_imgid2cid = [test_dataset[i][2] for i in range(eval_length)]
        eval_sampler = RandSubClassSampler(
            inds=range(eval_length),
            labels=eval_imgid2cid,
            cls_per_batch=test_cls_per_batch,
            batch_size=batch_size,
            num_batch=eval_length // batch_size)
        eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers, sampler=eval_sampler)
    else:
        eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=num_workers)
    eval_loader.num_classes = 169 

    return train_loader, eval_loader


def omniglot_alphabet_func(alphabet, background, root='../data'):
    def create_alphabet_dataset(batch_size, num_workers=2):
        # This dataset is only for unsupervised clustering
        # train_dataset (with data augmentation) is used during the optimization of clustering criteria
        # test_dataset (without data augmentation) is used after the clustering is converged

        binary_flip = transforms.Lambda(lambda x: 1 - x)
        normalize = transforms.Normalize((0.086,), (0.235,))

        train_dataset = Omniglot(
            root=root, download=True, background=background,
            transform=transforms.Compose(
                       [transforms.Resize(32),
                        transforms.ToTensor(),
                        binary_flip,
                        normalize]
                       ),
            deform=transforms.Compose([
                    transforms.RandomAffine(
                        degrees = (-5, 5), 
                        translate = (0.1, 0.1), 
                        scale = (0.8, 1.2), 
                        shear = (-10, 10), 
                        fillcolor = 255)
                                ])
                                )

        # Following part dependents on the internal implementation of official Omniglot dataset loader
        # Only use the images which has alphabet-name in their path name (_characters[cid])
        valid_flat_character_images = [(imgname,cid) for imgname,cid in train_dataset._flat_character_images if alphabet in train_dataset._characters[cid]]
        ndata = len(valid_flat_character_images)  # The number of data after filtering
        train_imgid2cid = [valid_flat_character_images[i][1] for i in range(ndata)]  # The tuple (valid_flat_character_images[i]) are (img, cid)
        cid_set = set(train_imgid2cid)  # The labels are not 0..c-1 here.
        cid2ncid = {cid:ncid for ncid,cid in enumerate(cid_set)}  # Create the mapping table for New cid (ncid)
        valid_characters = {cid2ncid[cid]:train_dataset._characters[cid] for cid in cid_set}
        for i in range(ndata):  # Convert the labels to make sure it has the value {0..c-1}
            valid_flat_character_images[i] = (valid_flat_character_images[i][0],cid2ncid[valid_flat_character_images[i][1]])

        # Apply surgery to the dataset
        train_dataset._flat_character_images = valid_flat_character_images
        train_dataset._characters = valid_characters

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers)
        train_loader.num_classes = len(cid_set)

        test_dataset = Omniglot(
            root=root, download=True, background=background,
            transform=transforms.Compose(
              [transforms.Resize(32),
               transforms.ToTensor(),
               binary_flip,
               normalize]
            ))
        # Apply surgery to the dataset
        test_dataset._flat_character_images = valid_flat_character_images  # Set the new list to the dataset
        test_dataset._characters = valid_characters

        eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers)
        eval_loader.num_classes = train_loader.num_classes

        print('=> Alphabet %s has %d characters and %d images.'%(alphabet, train_loader.num_classes, len(train_dataset)))
        return train_loader, eval_loader
    return create_alphabet_dataset

omniglot_background_alphabets=[
    'Alphabet_of_the_Magi', 
    'Gujarati',
    'Anglo-Saxon_Futhorc',  
    'Hebrew',  
    'Arcadian',  
    'Inuktitut_(Canadian_Aboriginal_Syllabics)',
    'Armenian',  
    'Japanese_(hiragana)',
    'Asomtavruli_(Georgian)',                     
    'Japanese_(katakana)',
    'Balinese',                                   
    'Korean',
    'Bengali',                                    
    'Latin',
    'Blackfoot_(Canadian_Aboriginal_Syllabics)',  
    'Malay_(Jawi_-_Arabic)',
    'Braille',                                    
    'Mkhedruli_(Georgian)',
    'Burmese_(Myanmar)',                          
    'N_Ko',
    'Cyrillic',                                   
    'Ojibwe_(Canadian_Aboriginal_Syllabics)',
    'Early_Aramaic',                              
    'Sanskrit',
    'Futurama',                                   
    'Syriac_(Estrangelo)',
    'Grantha',                                    
    'Tagalog',
    'Greek',                                     
    'Tifinagh'
        ]

omniglot_background_val_alphabets=[
    'Alphabet_of_the_Magi', 
    'Japanese_(katakana)',
    'Latin',
    'Cyrillic',                                   
    'Grantha'                                    
        ]

omniglot_evaluation_alphabets_mapping = {
    'Malayalam':'Malayalam',
     'Kannada':'Kannada',
     'Syriac':'Syriac_(Serto)',
     'Atemayar_Qelisayer':'Atemayar_Qelisayer',
     'Gurmukhi':'Gurmukhi',
     'Old_Church_Slavonic':'Old_Church_Slavonic_(Cyrillic)',
     'Manipuri':'Manipuri',
     'Atlantean':'Atlantean',
     'Sylheti':'Sylheti',
     'Mongolian':'Mongolian',
     'Aurek':'Aurek-Besh',
     'Angelic':'Angelic',
     'ULOG':'ULOG',
     'Oriya':'Oriya',
     'Avesta':'Avesta',
     'Tibetan':'Tibetan',
     'Tengwar':'Tengwar',
     'Keble':'Keble',
     'Ge_ez':'Ge_ez',
     'Glagolitic':'Glagolitic'
}

# Create the functions to access the individual alphabet dataset in Omniglot
for funcName, alphabetStr in omniglot_evaluation_alphabets_mapping.items():
    locals()['Omniglot_eval_' + funcName] = omniglot_alphabet_func(alphabet=alphabetStr, background=False)

def show_batch(inp, title=None):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg")
    """Show batch"""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()
    raw_input()

if __name__ == '__main__':
    import numpy as np
    train_loader, eval_loader = Omniglot_loader(batch_size=10, num_workers=2, root='./data_shallow14/datasets')
    print('len', len(train_loader.dataset), len(eval_loader.dataset))
    img, img_d, target, idx = next(iter(train_loader))
    
    print(target, idx)
    print(len(np.unique(target)))
    out = torchvision.utils.make_grid(img_d)
    show_batch(out, title=target)

