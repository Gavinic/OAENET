import numpy as np
import os
import torch
from data.dataloader import DataLoader
import matplotlib.pyplot as plt
from pandas import *
from tqdm import tqdm
from model.models_factory import ModelsFactory
from options.test_options import TestOptions


class Test(object):
    def __init__(self):
        self._opt = TestOptions().parse()
        self.data_dir = "./dataset/AffectNet/"
        self.model_path = './checkpoints/OAENet_ck+_ckpt.pth'
        self.phase = 'val'
        self.class_num = 7
        self.model = ModelsFactory().get_by_name(self._opt.model_name, self._opt).model
        self.model.load_state_dict(torch.load(self.model_path))
        self.dataloader = DataLoader(self._opt, train=False).load_data()['val']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.test()

    def test(self):
        print(os.path.join(self.model_path))
        self.model.eval()
        running_corrects = 0
        data_num = len(self.dataloader.dataset)
        print("data_num: ", data_num)
        step_num = 0
        confus_list = []

        for inputs, w_mask, mask_seg, labels in tqdm(self.dataloader):
            step_num += 1
            data_list = []
            inputs = inputs.to(self.device)
            w_mask = w_mask.to(self.device)
            mask_seg = mask_seg.to(self.device)
            labels = labels.to(self.device)
            self.model.to(self.device)

            outputs = self.model((inputs, w_mask))
            preds = torch.max(outputs, 1)[1]
            running_corrects += torch.sum(preds == labels.data)

            for i in range(self.class_num):
                tmp_list = []
                for j in range(self.class_num):
                    row_value = np.sum(np.array((labels.to('cpu').data == i) & (preds.to('cpu') == j)))
                    tmp_list.append(row_value)
                data_list.append(tmp_list)
            confus_list.append(data_list)

        print("confus_list:" + str(len(confus_list)) + '\n')

        result＿confu = np.zeros((self.class_num, self.class_num))
        for i in range(len(confus_list)):
            arr_1 = np.array(confus_list[i])
            result＿confu += arr_1
        print(result＿confu)

        tmp_value = result＿confu.tolist()
        row_value = list(map(sum, tmp_value))
        for i in range(len(result＿confu)):
            result＿confu[i] = result＿confu[i] / row_value[i] * 100
        print_result＿confu = np.around(result＿confu, decimals=2)
        print(print_result＿confu)

        epoch_acc = running_corrects.double() / data_num
        print('{} Acc: {:.4f}'.format(self.phase, epoch_acc))

    @staticmethod
    def show_batch(inputs, labels):
        plt.figure()
        print('input data shape: ', inputs.shape)
        print("inputs data: ", inputs[0])
        input_1 = inputs.detach().cpu().numpy()
        label_1 = labels.detach().cpu().numpy()
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.imshow(input_1[i].transpose([1, 2, 0]))
            print(label_1[i])
        plt.savefig('image.jpg')

    @staticmethod
    def precise_visual(confus_list, eva_dict, table_name):
        idx = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
        df = DataFrame(confus_list, index=idx,
                       columns=['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger'])
        vals = np.around(df.values, 2)

        fig = plt.figure(1, figsize=(15, 5))
        ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
        the_table = plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns,
                              colWidths=[0.1] * vals.shape[1], loc='center', cellLoc='center')
        the_table.set_fontsize(20)
        the_table.scale(2, 2.2)
        plt.savefig(table_name)


if __name__ == '__main__':
    print('.' * 50)
    Test()
