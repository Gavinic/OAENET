from data.dataset import ImageData
import torch


class DataLoader(object):
    def __init__(self, opt, train=True):
        if train:
            self.image_datasets = {
                x[0]: ImageData(opt.data_dir, train=x[1], img_size=opt.image_size)
                for x in [('train', True), ('val', False)]}
            self.dataloaders_dict = {
                x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.n_threads_train, pin_memory=False, drop_last=False)
                for x in ['train', 'val']}
        else:
            self.image_datasets = {
                x[0]: ImageData(opt.data_dir, train=x[1], img_size=opt.image_size)
                for x in [('val', False)]}
            self.dataloaders_dict = {
                x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.n_threads_train, pin_memory=False, drop_last=False)
                for x in ['val']}

    def load_data(self):
        return self.dataloaders_dict

    def __len__(self):
        return len(self.image_datasets)
