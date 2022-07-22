import numpy as np
import os
import time
import torch
from tensorboardX import SummaryWriter


class TBVisualizer:
    def __init__(self, opt):
        # self._save_path = os.path.join(opt.checkpoints_dir, opt.name)
        self._save_path = os.path.join('./log_events/', opt.tensornoard_name)
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
        self._log_path = os.path.join(self._save_path, 'loss_log.txt')
        self._tb_path = os.path.join(self._save_path, 'summary.json')
        self._writer = SummaryWriter(self._save_path)
        self.opt = opt

        with open(self._log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def __del__(self):
        self._writer.close()

    def _save_graph_(self, model, batch_size, input_size):
        # 在使用这个函数的时候一定要注意,将输入和模型放在同类型的设备上.tensorboardX 只能放在CPU上
        dummpy_input = torch.FloatTensor(batch_size, 3, input_size, input_size)
        # print(type(dummpy_input))
        self._writer.add_graph(model, dummpy_input.to(), True)

    def display_current_results(self, visuals, it, is_train, save_visuals=False):
        for label, image_numpy in visuals.items():
            sum_name = '{}/{}/{}'.format(self.opt.tensornoard_name, 'Train' if is_train else 'Test', label)

            # pytorch 输入的张量的shape为(C,H,W),tensorflow ,numpy 的都是(H,W,C)
            image_new = np.transpose(image_numpy, [2, 0, 1])
            # 将numpy 数组　转化成　pytorch的张量
            locs = torch.from_numpy(image_new)
            self._writer.add_image(sum_name, locs, it)

        self._writer.export_scalars_to_json(self._tb_path)

    def plot_scalars(self, scalars, it, is_train):
        for label, scalar in scalars.items():
            sum_name = '{}/{}/{}'.format(self.opt.tensornoard_name, 'Train' if is_train else 'epoch_val', label)
            self._writer.add_scalar(sum_name, scalar, it)

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t, visuals_were_stored):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        visuals_info = "v" if visuals_were_stored else ""
        message = '%s (T%s, epoch: %d, it: %d/%d, t/smpl: %.3fs)' % (
            log_time, visuals_info, epoch, i, iters_per_epoch, t)
        for k, v in errors.items():
            message += '%s:%.3f ' % (k, v)

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_validate_errors(self, epoch, errors, t):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (V, epoch: %d, time_to_val: %ds) ' % (log_time, epoch, t)
        for k, v in errors.items():
            message += '%s:%.3f ' % (k, v)

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)
