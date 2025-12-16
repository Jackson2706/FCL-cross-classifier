import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy, wandb
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from kornia import augmentation
import time, os, math
import torch.nn.init as init
from PIL import Image
import pickle
import shutil


class Buffer(nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        self.args = args
        self.k = 0.03

        self.place_left = True

        if input_size is None:
            input_size = args.input_size

        buffer_size = args.buffer_size
        print('buffer has %d slots' % buffer_size)

        bx = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        print("bx", bx.shape)
        by = torch.LongTensor(buffer_size).fill_(0)
        bt = torch.LongTensor(buffer_size).fill_(0)

        logits = torch.FloatTensor(buffer_size, args.n_classes).fill_(0)
        feature = torch.FloatTensor(buffer_size, 512).fill_(0)

        bx = bx.cuda()
        by = by.cuda()
        bt = bt.cuda()
        logits = logits.cuda()
        feature = feature.cuda()
        self.save_logits = None

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full = 0

        # registering as buffer allows us to save the object using torch.save
        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)
        self.register_buffer('logits', logits)
        self.register_buffer('feature', feature)
        self.to_one_hot = lambda x: x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x: torch.arange(x.size(0)).to(x.device)
        self.shuffle = lambda x: x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    @property
    def t(self):
        return self.bt[:self.current_index]

    @property
    def valid(self):
        return self.is_valid[:self.current_index]

    def display(self, gen=None, epoch=-1):
        from torchvision.utils import save_image
        from PIL import Image

        if 'cifar' in self.args.dataset:
            shp = (-1, 3, 32, 32)
        elif 'tinyimagenet' in self.args.dataset:
            shp = (-1, 3, 64, 64)
        else:
            shp = (-1, 1, 28, 28)

        if gen is not None:
            x = gen.decode(self.x)
        else:
            x = self.x

        save_image((x.reshape(shp) * 0.5 + 0.5), 'samples/buffer_%d.png' % epoch, nrow=int(self.current_index ** 0.5))
        # Image.open('buffer_%d.png' % epoch).show()
        print(self.y.sum(dim=0))

    def add_reservoir(self, x, y, logits, t):
        n_elem = x.size(0)
        save_logits = logits is not None
        self.save_logits = logits is not None

        # add whatever still fits in the bufferclass Buffer(nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        self.args = args
        self.k = 0.03

        self.place_left = True

        if input_size is None:
            input_size = args.input_size

        buffer_size = args.buffer_size
        print('buffer has %d slots' % buffer_size)

        bx = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        print("bx", bx.shape)
        by = torch.LongTensor(buffer_size).fill_(0)
        bt = torch.LongTensor(buffer_size).fill_(0)

        logits = torch.FloatTensor(buffer_size, args.n_classes).fill_(0)
        feature = torch.FloatTensor(buffer_size, 512).fill_(0)

        bx = bx.cuda()
        by = by.cuda()
        bt = bt.cuda()
        logits = logits.cuda()
        feature = feature.cuda()
        self.save_logits = None

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full = 0

        # registering as buffer allows us to save the object using torch.save
        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)
        self.register_buffer('logits', logits)
        self.register_buffer('feature', feature)
        self.to_one_hot = lambda x: x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x: torch.arange(x.size(0)).to(x.device)
        self.shuffle = lambda x: x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    @property
    def t(self):
        return self.bt[:self.current_index]

    @property
    def valid(self):
        return self.is_valid[:self.current_index]

    def display(self, gen=None, epoch=-1):
        from torchvision.utils import save_image
        from PIL import Image

        if 'cifar' in self.args.dataset:
            shp = (-1, 3, 32, 32)
        elif 'tinyimagenet' in self.args.dataset:
            shp = (-1, 3, 64, 64)
        else:
            shp = (-1, 1, 28, 28)

        if gen is not None:
            x = gen.decode(self.x)
        else:
            x = self.x

        save_image((x.reshape(shp) * 0.5 + 0.5), 'samples/buffer_%d.png' % epoch, nrow=int(self.current_index ** 0.5))
        # Image.open('buffer_%d.png' % epoch).show()
        print(self.y.sum(dim=0))

    def add_reservoir(self, x, y, logits, t):
        n_elem = x.size(0)
        save_logits = logits is not None
        self.save_logits = logits is not None

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)

            if save_logits:
                self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])
            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        self.place_left = False

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        assert idx_buffer.max() < self.bx.size(0)
        assert idx_buffer.max() < self.by.size(0)
        assert idx_buffer.max() < self.bt.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)

        # perform overwrite op
        self.bx[idx_buffer] = x[idx_new_data].cuda()
        self.by[idx_buffer] = y[idx_new_data].cuda()
        self.bt[idx_buffer] = t

        if save_logits:
            self.logits[idx_buffer] = logits[idx_new_data]

        return idx_buffer

    def measure_valid(self, generator, classifier):
        with torch.no_grad():
            # fetch valid examples
            valid_indices = self.valid.nonzero()
            valid_x, valid_y = self.bx[valid_indices], self.by[valid_indices]
            one_hot_y = self.to_one_hot(valid_y.flatten())

            hid_x = generator.idx_2_hid(valid_x)
            x_hat = generator.decode(hid_x)

            logits = classifier(x_hat)
            _, pred = logits.max(dim=1)
            one_hot_pred = self.to_one_hot(pred)
            correct = one_hot_pred * one_hot_y

            per_class_correct = correct.sum(dim=0)
            per_class_deno = one_hot_y.sum(dim=0)
            per_class_acc = per_class_correct.float() / per_class_deno.float()
            self.class_weight = 1. - per_class_acc
            self.valid_acc = per_class_acc
            self.valid_deno = per_class_deno

    def shuffle_(self):
        indices = torch.randperm(self.current_index).to(self.args.device)
        self.bx = self.bx[indices]
        self.by = self.by[indices]
        self.bt = self.bt[indices]
        
    class Buffer(nn.Module):
        def __init__(self, args, input_size=None):
            super().__init__()
            self.args = args
            self.k = 0.03

            self.place_left = True

            if input_size is None:
                input_size = args.input_size

            buffer_size = args.buffer_size
            print('buffer has %d slots' % buffer_size)

            bx = torch.FloatTensor(buffer_size, *input_size).fill_(0)
            print("bx", bx.shape)
            by = torch.LongTensor(buffer_size).fill_(0)
            bt = torch.LongTensor(buffer_size).fill_(0)

            logits = torch.FloatTensor(buffer_size, args.n_classes).fill_(0)
            feature = torch.FloatTensor(buffer_size, 512).fill_(0)

            bx = bx.cuda()
            by = by.cuda()
            bt = bt.cuda()
            logits = logits.cuda()
            feature = feature.cuda()
            self.save_logits = None

            self.current_index = 0
            self.n_seen_so_far = 0
            self.is_full = 0

            # registering as buffer allows us to save the object using torch.save
            self.register_buffer('bx', bx)
            self.register_buffer('by', by)
            self.register_buffer('bt', bt)
            self.register_buffer('logits', logits)
            self.register_buffer('feature', feature)
            self.to_one_hot = lambda x: x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
            self.arange_like = lambda x: torch.arange(x.size(0)).to(x.device)
            self.shuffle = lambda x: x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    @property
    def t(self):
        return self.bt[:self.current_index]

    @property
    def valid(self):
        return self.is_valid[:self.current_index]

    def display(self, gen=None, epoch=-1):
        from torchvision.utils import save_image
        from PIL import Image

        if 'cifar' in self.args.dataset:
            shp = (-1, 3, 32, 32)
        elif 'tinyimagenet' in self.args.dataset:
            shp = (-1, 3, 64, 64)
        else:
            shp = (-1, 1, 28, 28)

        if gen is not None:
            x = gen.decode(self.x)
        else:
            x = self.x

        save_image((x.reshape(shp) * 0.5 + 0.5), 'samples/buffer_%d.png' % epoch, nrow=int(self.current_index ** 0.5))
        # Image.open('buffer_%d.png' % epoch).show()
        print(self.y.sum(dim=0))

    def add_reservoir(self, x, y, logits, t):
        n_elem = x.size(0)
        save_logits = logits is not None
        self.save_logits = logits is not None

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)

            if save_logits:
                self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])
            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        self.place_left = False

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        assert idx_buffer.max() < self.bx.size(0)
        assert idx_buffer.max() < self.by.size(0)
        assert idx_buffer.max() < self.bt.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)

        # perform overwrite op
        self.bx[idx_buffer] = x[idx_new_data].cuda()
        self.by[idx_buffer] = y[idx_new_data].cuda()
        self.bt[idx_buffer] = t

        if save_logits:
            self.logits[idx_buffer] = logits[idx_new_data]

        return idx_buffer

    def measure_valid(self, generator, classifier):
        with torch.no_grad():
            # fetch valid examples
            valid_indices = self.valid.nonzero()
            valid_x, valid_y = self.bx[valid_indices], self.by[valid_indices]
            one_hot_y = self.to_one_hot(valid_y.flatten())

            hid_x = generator.idx_2_hid(valid_x)
            x_hat = generator.decode(hid_x)

            logits = classifier(x_hat)
            _, pred = logits.max(dim=1)
            one_hot_pred = self.to_one_hot(pred)
            correct = one_hot_pred * one_hot_y

            per_class_correct = correct.sum(dim=0)
            per_class_deno = one_hot_y.sum(dim=0)
            per_class_acc = per_class_correct.float() / per_class_deno.float()
            self.class_weight = 1. - per_class_acc
            self.valid_acc = per_class_acc
            self.valid_deno = per_class_deno

    def shuffle_(self):
        indices = torch.randperm(self.current_index).to(self.args.device)
        self.bx = self.bx[indices]
        self.by = self.by[indices]
        self.bt = self.bt[indices]

    def delete_up_to(self, remove_after_this_idx):
        self.bx = self.bx[:remove_after_this_idx]
        self.by = self.by[:remove_after_this_idx]
        self.br = self.bt[:remove_after_this_idx]


    def delete_up_to(self, remove_after_this_idx):
        self.bx = self.bx[:remove_after_this_idx]
        self.by = self.by[:remove_after_this_idx]
        self.br = self.bt[:remove_after_this_idx]

        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)

            if save_logits:
                self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])
            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        self.place_left = False

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        assert idx_buffer.max() < self.bx.size(0)
        assert idx_buffer.max() < self.by.size(0)
        assert idx_buffer.max() < self.bt.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)

        # perform overwrite op
        self.bx[idx_buffer] = x[idx_new_data].cuda()
        self.by[idx_buffer] = y[idx_new_data].cuda()
        self.bt[idx_buffer] = t

        if save_logits:
            self.logits[idx_buffer] = logits[idx_new_data]

        return idx_buffer

    def measure_valid(self, generator, classifier):
        with torch.no_grad():
            # fetch valid examples
            valid_indices = self.valid.nonzero()
            valid_x, valid_y = self.bx[valid_indices], self.by[valid_indices]
            one_hot_y = self.to_one_hot(valid_y.flatten())

            hid_x = generator.idx_2_hid(valid_x)
            x_hat = generator.decode(hid_x)

            logits = classifier(x_hat)
            _, pred = logits.max(dim=1)
            one_hot_pred = self.to_one_hot(pred)
            correct = one_hot_pred * one_hot_y

            per_class_correct = correct.sum(dim=0)
            per_class_deno = one_hot_y.sum(dim=0)
            per_class_acc = per_class_correct.float() / per_class_deno.float()
            self.class_weight = 1. - per_class_acc
            self.valid_acc = per_class_acc
            self.valid_deno = per_class_deno

    def shuffle_(self):
        indices = torch.randperm(self.current_index).to(self.args.device)
        self.bx = self.bx[indices]
        self.by = self.by[indices]
        self.bt = self.bt[indices]

    def delete_up_to(self, remove_after_this_idx):
        self.bx = self.bx[:remove_after_this_idx]
        self.by = self.by[:remove_after_this_idx]
        self.br = self.bt[:remove_after_this_idx]

    def sample(self, amt, exclude_task=None, ret_ind=False):
        if self.save_logits:
            if exclude_task is not None:
                valid_indices = (self.t != exclude_task)
                valid_indices = valid_indices.nonzero().squeeze()
                bx, by, bt, logits = self.bx[valid_indices], self.by[valid_indices], self.bt[valid_indices], \
                                     self.logits[valid_indices]
            else:
                bx, by, bt, logits = self.bx[:self.current_index], self.by[:self.current_index], \
                    self.bt[:self.current_index], self.logits[:self.current_index]

            if bx.size(0) < amt:
                if ret_ind:
                    return bx, by, logits, bt, torch.from_numpy(np.arange(bx.size(0)))
                else:
                    return bx, by, logits, bt
            else:
                indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False))

                indices = indices.cuda()

                if ret_ind:
                    return bx[indices], by[indices], logits[indices], bt[indices], indices
                else:
                    return bx[indices], by[indices], logits[indices], bt[indices]
        else:
            if exclude_task is not None:
                valid_indices = (self.t != exclude_task)
                valid_indices = valid_indices.nonzero().squeeze()
                bx, by, bt = self.bx[valid_indices], self.by[valid_indices], self.bt[valid_indices]
            else:
                bx, by, bt = self.bx[:self.current_index], self.by[:self.current_index], self.bt[:self.current_index]

            if bx.size(0) < amt:
                if ret_ind:
                    return bx, by, bt, torch.from_numpy(np.arange(bx.size(0)))
                else:
                    return bx, by, bt
            else:
                indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False)).long()

                indices = indices.cuda()

                if ret_ind:
                    return bx[indices], by[indices], bt[indices], indices
                else:
                    return bx[indices], by[indices], bt[indices]

    def split(self, amt):
        indices = torch.randperm(self.current_index).to(self.args.device)
        return indices[:amt], indices[amt:]

    def onlysample(self, amt, task=None, ret_ind=False):

        if self.save_logits:
            if task is not None:
                valid_indices = (self.t == task)
                valid_indices = valid_indices.nonzero().squeeze()
                bx, by, bt, logits = self.bx[valid_indices], self.by[valid_indices], self.bt[valid_indices], \
                                     self.logits[valid_indices]
            else:
                bx, by, bt, logits = self.bx[:self.current_index], self.by[:self.current_index], \
                    self.bt[:self.current_index], self.logits[:self.current_index]

            if bx.size(0) < amt:
                if ret_ind:
                    return bx, by, logits, bt, torch.from_numpy(np.arange(bx.size(0)))
                else:
                    return bx, by, logits, bt
            else:
                indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False))

                indices = indices.cuda()

                if ret_ind:
                    return bx[indices], by[indices], logits[indices], bt[indices], indices
                else:
                    return bx[indices], by[indices], logits[indices], bt[indices]
        else:
            return 0


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join(self.root, "%d.png" % (self._idx)), pack=False)
        self._idx += 1

    def get_dataset(self, nums=None, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform, nums=nums)


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + '-%d.png' % (idx))


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(nums, self.root)  # [ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        # print("1: %s" % str(np.array(img).shape))
        if self.transform:
            img = self.transform(img)
        # print("2: %s" % str(img.shape))
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
        self.root, len(self), self.transform)


def _collect_all_images(nums, root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            if nums != None:
                files.sort()
                files = files[:nums]
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join(dirpath, f))
    return images