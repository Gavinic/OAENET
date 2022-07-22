import cv2
import math
import numpy as np
from skimage import feature as ft
from PIL import Image

np.seterr(divide='ignore', invalid='ignore')


class Weight_Mask:
    def __init__(self, N=4, K=8, M=15, img_size=224):
        self.M = M
        self.N = N
        self.K = K
        self.img_size = img_size
        self.distance_matrix = self.get_distance_matrix(M=M)

    def get_distance_matrix(self, M=15):
        assert M % 2 == 1
        M_c = M // 2
        DM = np.ones((M, M, 3))
        for row in range(M):
            for col in range(M):
                DM[row, col, :] = 1 - abs(row - M_c) / M - abs(col - M_c) / M
        return DM

    def oriented_gradient(self, img, multichannel=True):
        img_t = img.copy()
        if multichannel:
            features = ft.hog(img_t, orientations=self.K, pixels_per_cell=(224, 56), cells_per_block=(1, 1),
                              visualize=False, channel_axis=2, feature_vector=True, block_norm='L1')
            features2 = ft.hog(img_t, orientations=self.K, pixels_per_cell=(224, 224), cells_per_block=(1, 1),
                               visualize=False, channel_axis=2, feature_vector=True, block_norm='L1')
        else:
            img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
            features = ft.hog(img_t, orientations=self.K, pixels_per_cell=(224, 56), cells_per_block=(1, 1),
                              visualize=False, channel_axis=None, feature_vector=True, block_norm='L1')
            features2 = ft.hog(img_t, orientations=self.K, pixels_per_cell=(224, 224), cells_per_block=(1, 1),
                               visualize=False, channel_axis=None, feature_vector=True, block_norm='L1')
        features = features.reshape(-1, 8)
        rou = np.zeros((4))
        for i in range(4):
            rou_t = np.corrcoef(features[i], features2)[0, 1]
            if rou_t <= 1 and rou_t >= -1:
                rou[i] = rou_t
            else:
                rou[i] = 1
        return rou

    def get_weight_Mask(self, img, lmk):
        M = self.M
        rou = self.oriented_gradient(img)
        sub_h = img.shape[0] // 4

        mask = np.zeros(img.shape, dtype=float)
        pad = 7
        mask_pad = np.ones((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad, img.shape[2]), dtype=float)
        for i in range(self.N):
            mask[i * sub_h:(i + 1) * sub_h, :] = rou[i]

        mask_pad[pad:img.shape[0] + pad, pad:img.shape[0] + pad, :] = mask
        for point in lmk:
            x, y = (int(point[0]), int(point[1]))
            yt, yd, xl, xr = (y - M // 2 + pad, y + M // 2 + 1 + pad, x - M // 2 + pad, x + M // 2 + 1 + pad)
            if yt < 0 or xl < 0 or yd >= (self.img_size + 2 * pad) or xr >= (self.img_size + 2 * pad):
                continue
            mask_pad[yt:yd, xl:xr, :] = self.distance_matrix
        mask = mask_pad[pad:img.shape[0] + pad, pad:img.shape[0] + pad, :]
        return mask


def corp_face(image_array, landmarks):
    eye_center = np.mean(landmarks[36:48], axis=0).astype("int")
    lip_center = np.mean(landmarks[48:68], axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = bottom - top
    x_min = np.min(landmarks, axis=0)[0]
    x_max = np.max(landmarks, axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    landmarks[:, 0] -= left
    landmarks[:, 1] -= top
    return cropped_img, landmarks


def align_face(image_array, lmk):
    # get list landmarks of left and right eye
    left_eye = lmk[36:42]
    right_eye = lmk[42:48]
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)

    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                  (left_eye_center[1] + right_eye_center[1]) / 2)

    # at the eye_center, rotate the image by the angle
    rotate = cv2.getRotationMatrix2D((round(eye_center[0]), round(eye_center[1])), angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate, (image_array.shape[1], image_array.shape[0]))
    lm_raw = lmk.copy()
    lm_raw[:, 0] = lmk[:, 0] * rotate[0, 0] + lmk[:, 1] * rotate[0, 1] + rotate[0, 2]
    lm_raw[:, 1] = lmk[:, 0] * rotate[1, 0] + lmk[:, 1] * rotate[1, 1] + rotate[1, 2]
    return rotated_img, lm_raw
