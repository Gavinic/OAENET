import argparse
import torch

class BaseOptions(object):
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--data_dir', type=str, help='path to dataset')
        self._parser.add_argument('--csv_path', type=str, help='path to csv_path')
        self._parser.add_argument('--tensornoard_name', type=str, help='path to tensorboardx log')
        self._parser.add_argument('--log_interval', type=int, default=100, help='show the progress')
        self._parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self._parser.add_argument('--num_classes', type=int, default=7, help='calss number')
        self._parser.add_argument('--image_size', type=int, default=128, help='input image size')
        self._parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self._parser.add_argument('--model_name', type=str, default='ganimation', help='model to run[au_net_model]')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or set
        self._opt.is_train = self.is_train

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        return self._opt

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

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
