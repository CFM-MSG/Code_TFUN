from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import sys
import pickle
import numpy as np
import lmdb
import torch

def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        print(..., file=sys.stderr)
        return Image.new('RGB', (224, 224), 'white')
       
class ImagerLoader(data.Dataset):
    def __init__(self, img_path, transform=None, target_transform=None,
                 loader=default_loader, square=False, data_path=None, partition=None, sem_reg=None):

        if data_path is None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition

        self.env = lmdb.open(os.path.join(data_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with open(os.path.join(data_path, partition + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)  # indices of all images

        self.square = square
        self.imgPath = img_path
        self.maxInst = 20

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        # print(index)
        recipId = self.ids[index]
        # we only need match pair
        if self.partition == 'train' or self.partition == 'val' or self.partition == 'test':
            match = True
        else:
            raise Exception('Partition name not well defined')

        target = match and 1 or -1

        with self.env.begin(write=False) as txn:
                serialized_sample = txn.get(self.ids[index].encode('latin1'))
        try:
            sample = pickle.loads(serialized_sample,encoding='latin1')
        except ValueError:
            return None

        imgs = sample['imgs']

        # # same dishes in a sample, sample are decided by index

        # image
        #if target == 1:
        if self.partition == 'train' or self.partition == 'test':
            # We do only use the one of the first five images per recipe during training
            imgIdx = np.random.choice(range(min(5, len(imgs))))
        else:
            imgIdx = 0
        loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
        loader_path = os.path.join(*loader_path)
        path = os.path.join(self.imgPath, self.partition, loader_path, imgs[imgIdx]['id'])
        # path = os.path.join(self.imgPath, loader_path, imgs[imgIdx]['id'])

        # instructions
        instrs = sample['intrs']
        itr_ln = len(instrs)
        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)  # skip-instructions vectors

        # ingredients
        ingrs = sample['ingrs'].astype(int)  # ingredients ids
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

        # load image
        img = self.loader(path)

        if self.square:
            img = img.resize(self.square)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        rec_class = sample['classes'] - 1
        rec_id = self.ids[index]

        img_class = sample['classes'] - 1
        img_id = self.ids[index]


        # output
        if self.partition == 'train' or self.partition == 'test':
            return [img, instrs, itr_ln, ingrs, igr_ln], [target, index, len(imgs)]
        else:
            return [img, instrs, itr_ln, ingrs, igr_ln], [target, img_id, rec_id]

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)