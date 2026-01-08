import copy
import time

import torch
from flcore.clients.clientbase import Client
from torch.utils.data import DataLoader, Dataset


class ReplayDataset(Dataset):
    def __init__(self, real_dataset, syn_images, syn_labels):
        self.real_dataset = real_dataset
        self.syn_images = syn_images
        self.syn_labels = syn_labels
        self.real_len = len(real_dataset)

    def __len__(self):
        return self.real_len + len(self.syn_labels)

    def __getitem__(self, index):
        if index < self.real_len:
            # Return real data: (img, label) or (img, label, idx)
            item = self.real_dataset[index]
            # Ensure consistent return length (image, label, dummy_index)
            img, label = item[0], item[1]
            return img, label, index
        else:
            syn_idx = index - self.real_len
            img = self.syn_images[syn_idx]
            label = self.syn_labels[syn_idx]
            # Synthetic index is flagged as -1
            return img, label, -1

class clientOursV2(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        self.img_size = 32 if 'cifar' in args.dataset.lower() else 224
        self.nz = 256 if 'cifar100' in args.dataset.lower() else 100
        self.generator = None

    def set_generator_parameters(self, global_generator):
        self.generator = copy.deepcopy(global_generator).to(self.device).eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def train(self, task=None):
        train_loader = self.load_train_data(task=task)
        self.real_classes = set(self.current_labels)
        
        # Augment with synthetic replay
        augmented_loader = self.create_augmented_dataset(train_loader.dataset)
        
        self.model.train()
        start_time = time.time()
        for epoch in range(self.local_epochs):
            for batch in augmented_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss(self.model(x), y)
                loss.backward()
                self.optimizer.step()
            self.learning_rate_scheduler.step()
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def create_augmented_dataset(self, train_data):
        if self.generator is None or self.current_task == 0:
            return DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        # Replay classes from previous tasks
        print(f"Client {self.id}: Augmenting data with synthetic samples for missing classes.")
        missing_classes = list(set(self.classes_so_far) - self.real_classes)
        if not missing_classes:
            return DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        samples_per_class = max(10, len(train_data) // (len(missing_classes) + 1))
        all_syn_imgs, all_syn_labels = [], []

        for cid in missing_classes:
            z = torch.randn(samples_per_class, self.nz).to(self.device)
            lbls = torch.full((samples_per_class,), cid, dtype=torch.long).to(self.device)
            with torch.no_grad():
                all_syn_imgs.append(self.generator(z, lbls).cpu())
                all_syn_labels.append(lbls.cpu())

        syn_imgs = torch.cat(all_syn_imgs, dim=0)
        syn_lbls = torch.cat(all_syn_labels, dim=0)
        
        return DataLoader(ReplayDataset(train_data, syn_imgs, syn_lbls), batch_size=self.batch_size, shuffle=True)