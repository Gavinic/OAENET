import torch.nn as nn
import copy
import torch
import torch.optim as optim
import time
import os
import utils.utils as utils
from utils.ohem import OHEM_Loss, OAELoss
from collections import OrderedDict
from utils.tensob_visual import TBVisualizer
from model.models_factory import ModelsFactory
from data.dataloader import DataLoader
from torch.optim import lr_scheduler
from options.train_options import TrainOptions
import matplotlib.pyplot as plt


class Train:
    def __init__(self):

        self._opt = TrainOptions().parse()
        self.model = ModelsFactory().get_by_name(self._opt.model_name, self._opt).model
        self.model.init_weights()
        self._gpu_ids = self._opt.gpu_ids
        self.dataloader = DataLoader(self._opt).load_data()
        self.params_to_update = self.model.parameters()
        self.optimizer_ft = optim.SGD(self.params_to_update, lr=self._opt.learning_rate, momentum=self._opt.momentum)
        self.val_criterion = nn.CrossEntropyLoss()
        self.criterion = OAELoss(weight=None)
        self._tb_visualizer = TBVisualizer(self._opt)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if len(self._gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self._gpu_ids)
        self.model.to(self.device)
        self.total_steps_train = 0
        self.total_steps_val = 0
        self.total_train_loss = 0
        self.total_train_correct = 0
        self.total_val_loss = 0
        self.total_val_correct = 0
        self.train_model()

    def train_model(self):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        data_number_train = len(self.dataloader['train'].dataset)
        data_number_val = len(self.dataloader['val'].dataset)

        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer_ft, 'min', factor=0.1, patience=7)
        for epoch in range(self._opt.num_epochs):
            print('Epoch {}/{}'.format(epoch, self._opt.num_epochs - 1))
            print('-' * 50)

            total_loss_train, total_logists_train = self.train_epoch('train', True)
            epoch_loss_train = total_loss_train / data_number_train
            epoch_acc_train = total_logists_train.double() / data_number_train
            print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss_train, epoch_acc_train))

            total_loss_val, total_logists_val = self.val_epoch('val')
            epoch_loss_val = total_loss_val / data_number_val
            epoch_acc_val = total_logists_val.double() / data_number_val
            print('{} Loss: {:.4f} Acc: {:.4f}'.format('Val', epoch_loss_val, epoch_acc_val))

            scheduler.step(epoch_loss_val)
            if epoch_acc_val > best_acc:
                best_acc = epoch_acc_val
                best_model_wts = copy.deepcopy(self.model.module.state_dict())

            self._tb_visualizer._writer.add_scalars(self._opt.model_name + '/epoch_acc',
                                                    {'train_acc': epoch_acc_train, 'val_acc': epoch_acc_val}, epoch)

            self._tb_visualizer._writer.add_scalars(self._opt.model_name + '/epoch_loss',
                                                    {'train_loss': epoch_loss_train, 'val_loss': epoch_loss_val}, epoch)

            if not os.path.exists(self._opt.checkpoints_dir):
                os.makedirs(self._opt.checkpoints_dir)
            torch.save(self.model.module.state_dict(), os.path.join(self._opt.checkpoints_dir,
                                                                    'cate_5_Res34_' + str(epoch) + '.pth'))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        torch.save(best_model_wts, os.path.join(self._opt.checkpoints_dir, 'cate_5_best_model.pth'))

    def train_epoch(self, phase, keep_data_for_visual):
        print(phase, ' >>>>>>>>>>>>>>>>>>>>>>>>>')
        epoch_total_step = 0
        running_loss = 0.0
        running_corrects = 0
        self.model.train()
        for batch_idx, (image, w_mask, mask_seg, labels) in enumerate(self.dataloader[phase]):
            self.total_steps_train += 1
            epoch_total_step += 1
            image, w_mask, mask_seg, labels = self.set_input(image, w_mask, mask_seg, labels)
            self.optimizer_ft.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model((image, w_mask))
                loss = self.criterion(outputs, labels, mask_seg)
                preds = torch.max(outputs[0], 1)[1]
                if phase == 'train':
                    loss.backward()
                    self.optimizer_ft.step()

            running_loss += loss.item() * image.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if self.total_steps_train % 100 == 0:
                current_iters = image.size(0) * epoch_total_step
                accuracy_steps = running_corrects.double() / current_iters
                loss_steps = running_loss / current_iters

                sum_name = '{}/{}/{}'.format(self._opt.model_name, 'Train', 'loss')
                self._tb_visualizer._writer.add_scalar(sum_name, loss_steps, self.total_steps_train)
                sum_name = '{}/{}/{}'.format(self._opt.model_name, 'Train', 'accuracy')
                self._tb_visualizer._writer.add_scalar(sum_name, accuracy_steps, self.total_steps_train)

            if self.total_steps_train % 1000 == 0:
                self._tb_visualizer.display_current_results(self.get_current_visuals(), self.total_steps_train,
                                                            is_train=True)

            if phase == 'train' and batch_idx % self._opt.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * image.size(0), len(self.dataloader[phase].dataset),
                    100. * batch_idx * image.size(0) / len(self.dataloader[phase].dataset), loss.item())
                print(message)

            if keep_data_for_visual:
                self._vis_real_img = utils.tensor2im(image, idx=-1)

        return running_loss, running_corrects

    def val_epoch(self, phase):
        print(phase, ' >>>>>>>>>>>>>>>>>>>>>>>>>', )
        running_loss = 0.0
        running_corrects = 0
        self.model.eval()
        epoch_total_step = 0



        for batch_idx, (image, w_mask, mask_seg, labels) in enumerate(self.dataloader[phase]):
            epoch_total_step += 1
            image, w_mask, mask_seg, labels = self.set_input(image, w_mask, mask_seg, labels)
            self.optimizer_ft.zero_grad()
            torch.set_grad_enabled(False)
            outputs = self.model((image, w_mask))
            loss = self.val_criterion(outputs, labels)
            preds = torch.max(outputs, 1)[1]

            self.total_steps_val += 1

            running_loss += loss.item() * image.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # 验证集合上每10个step绘制一次loss以及accuracy
            if self.total_steps_val % 10 == 0:
                # batch_correct = torch.sum(preds == labels.data)
                current_iters = image.size(0) * epoch_total_step
                accuracy_steps = running_corrects.double() / current_iters
                loss_steps = running_loss / current_iters
                sum_name = '{}/{}/{}'.format(self._opt.model_name, 'val', 'loss')
                self._tb_visualizer._writer.add_scalar(sum_name, loss_steps, self.total_steps_val)
                sum_name = '{}/{}/{}'.format(self._opt.model_name, 'val', 'accuracy')
                self._tb_visualizer._writer.add_scalar(sum_name, accuracy_steps, self.total_steps_val)

        return running_loss, running_corrects

    # def get_current_loss(self):
    #     loss_dict = OrderedDict([('accuracy',self.epoch_acc),
    #                              ('loss', self.epoch_loss),
    #                              ('learning_rate',self.learning_rate)])
    #     return loss_dict

    def get_current_visuals(self):
        """ show images """
        visuals = OrderedDict()
        visuals['input_image_0'] = self._vis_real_img

        return visuals

    def show_batch(self, inputs, labels):
        plt.figure()
        print('input data shape: ', inputs.shape)
        input_1 = inputs.detach().cpu().numpy()
        label_1 = labels.detach().cpu().numpy()
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.imshow(input_1[i].transpose([1, 2, 0]))
            print(label_1[i])
        # plt.show()
        plt.savefig('image_2.jpg')

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

        # set gpu ids
        if len(self._opt.gpu_ids) > 0:
            torch.cuda.set_device(self._opt.gpu_ids[0])

    def set_input(self, inputs, w_mask, mask_seg, labels):
        if len(self._gpu_ids) > 0:
            inputs = inputs.cuda(self._gpu_ids[0], non_blocking=True)
            w_mask = w_mask.cuda(self._gpu_ids[0], non_blocking=True)
            mask_seg = mask_seg.cuda(self._gpu_ids[0], non_blocking=True)
            labels = labels.cuda(self._gpu_ids[0], non_blocking=True)
        return inputs, w_mask, mask_seg, labels


if __name__ == '__main__':
    Train()
