import numpy as np
import cv2
import os
import math
import torchvision


def file_clean(file_path_ori, file_path_new):
    '''处理原始的数据,得到三类的数据及label信息'''
    file_list_new = []
    with open(file_path_ori, 'r') as file_object:
        file_list = file_object.readlines()
        print('file_num before clean: ', len(file_list))
        for file in file_list:
            # print(file)
            label = file.split(' ')[1].split('\n')[0]
            '''指定所需要的label'''
            if label != '8':
                file_list_new.append(file)
    print('file_list_new:' + str(len(file_list_new)))
    with open(file_path_new, 'w') as file_object:
        for file in file_list_new:
            # print(file)
            file_object.write(file)

'''
def file_clean_bj(file_path_ori, file_path_new):
    #处理原始的数据,得到7类的数据及label信息
    file_list_new = []
    with open(file_path_ori, 'r') as file_object:
        file_list = file_object.readlines()
        print(len(file_list))
        count = 0
        for file in file_list:
            # print(file)
            label = file.split(' ')[1].split('\n')[0]

            if label == '2' or label == '3':
                file_list_new.append(file)

            if label == '7':
                count += 1
                if count <= 1000:
                    file_list_new.append(file)
                else:
                    continue
    print(count)
    print('file_list_new:' + str(len(file_list_new)))
    with open(file_path_new, 'w') as file_object:
        for file in file_list_new:
            # print(file)
            file_object.write(file)
'''



def convert_cv2numpy(img, img_list):
    '''将图像数据转化为numpy数组数据进行存储'''
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    img_list.append(img2)


def read_valid_img(img_path):
    '''图像数据标准化'''
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    return img


def read_face_data(img_list, img_root_path, flag):
    '''读入全脸以及部分器官图片数据'''
    # print(img_root_path)
    left_eye_imgs = []
    nose_imgs = []
    mouth_imgs = []
    whole_face_imgs = []
    labels = []

    cout_i = 0
    for value in img_list:
        if cout_i % 1000 == 0:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(cout_i)
        cout_i += 1
        if flag == 'raf':
            # raf_db 图像名称有_aligned
            img_name = value.split(" ")[0] + '_aligned' + '.jpg'
        if flag == 'bj':
            # 北京数据读取
            img_name = value.split(" ")[0] + '.jpg'
        if flag == 'aug':
            # 增强数据读取
            img_name = value.split(" ")[0]

        if os.path.exists(img_root_path + "left_eye/" + img_name) != True or \
                os.path.exists(img_root_path + "nose/" + img_name) != True or \
                os.path.exists(img_root_path + "whole_face/" + img_name) != True or \
                os.path.exists(img_root_path + "mouth/" + img_name) != True:
            continue  # 这里的continue的作用是跳过本次循环continue之后的语句

        if os.path.getsize(img_root_path + "left_eye/" + img_name) == 0 or \
                os.path.getsize(img_root_path + "nose/" + img_name) == 0 or \
                os.path.getsize(img_root_path + "whole_face/" + img_name) == 0 or \
                os.path.getsize(img_root_path + "mouth/" + img_name) == 0:
            continue

        # 将图像进行缩放,变成100*100
        left_eye_img = read_valid_img(img_root_path + "left_eye/" + img_name)
        nose_img = read_valid_img(img_root_path + "nose/" + img_name)
        mouth_img = read_valid_img(img_root_path + "mouth/" + img_name)
        whole_face_img = read_valid_img(img_root_path + "whole_face/" + img_name)

        # 将图像通道存储顺序变换为rgb
        convert_cv2numpy(left_eye_img, left_eye_imgs)
        convert_cv2numpy(nose_img, nose_imgs)
        convert_cv2numpy(mouth_img, mouth_imgs)
        convert_cv2numpy(whole_face_img, whole_face_imgs)

        print(img_root_path + "whole_face/" + img_name)
        # 分离得到图像标签
        label = value.split(" ")[1]
        labels.append(int(label))

    left_eye_data = np.array(left_eye_imgs)
    nose_data = np.array(nose_imgs)
    mouth_data = np.array(mouth_imgs)
    whole_face_data = np.array(whole_face_imgs)

    print('left_eye_shape: ' + str(left_eye_data.shape))
    print('nose_data_shape: ' + str(nose_data.shape))
    print('mouth_data_shape:' + str(mouth_data.shape))
    print('whole_face_data_shape: ' + str(whole_face_data.shape))
    print('labels: ' + str(len(labels)))

    return (left_eye_data, nose_data, mouth_data, whole_face_data, labels)


def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*254.0

    return image_numpy_t.astype(imtype)

def tensor2maskim(mask, imtype=np.uint8, idx=0, nrows=1):
    im = tensor2im(mask, imtype=imtype, idx=idx, unnormalize=False, nrows=nrows)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=-1)
    return im
