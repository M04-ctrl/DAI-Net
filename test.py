#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from models.factory import build_net
from torchvision.utils import make_grid
import glob
import warnings; warnings.filterwarnings('ignore', category=UserWarning)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_dtype(torch.float32)   # �޸� set_default_tensor_type ��������
    torch.set_default_device('cuda')
    cudnn.benchmark = True                   # �޸�ƴд���� benckmark �� benchmark
else:
    torch.set_default_dtype(torch.float32)


def tensor_to_image(tensor):
    grid = make_grid(tensor)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr

def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    image = image[[2, 1, 0], :, :]
    return image

def detect_face(img, tmp_shrink):
    image = cv2.resize(img, None, None, fx=tmp_shrink,
                       fy=tmp_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x = x / 255.
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))

    if use_cuda:
        x = x.cuda()

    y = net.test_forward(x)[0]
    detections = y.data.cpu().numpy()
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    boxes = []
    scores = []
    for i in range(detections.shape[1]):
        j = 0
        while ((j < detections.shape[2]) and detections[0, i, j, 0] > 0.0):
            pt = (detections[0, i, j, 1:] * scale)
            score = detections[0, i, j, 0]
            boxes.append([pt[0], pt[1], pt[2], pt[3]])
            scores.append(score)
            j += 1

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0, 0, 0, 0, 0.001]])

    det_xmin = boxes[:, 0]
    det_ymin = boxes[:, 1]
    det_xmax = boxes[:, 2]
    det_ymax = boxes[:, 3]
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    return det


def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(image, max_im_shrink):
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s, detect_face(image, 0.75)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt)

    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b, detect_face(image, 1.5)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(image, max_im_shrink)))

    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def multi_scale_test_pyramid(image, max_shrink):
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if st[i] <= max_shrink:
            det_temp = detect_face(image, st[i])
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b


def bbox_vote(det_):
    order_ = det_[:, 4].ravel().argsort()[::-1]
    det_ = det_[order_, :]
    dets_ = np.zeros((0, 5), dtype=np.float32)
    while det_.shape[0] > 0:
        area_ = (det_[:, 2] - det_[:, 0] + 1) * (det_[:, 3] - det_[:, 1] + 1)
        xx1 = np.maximum(det_[0, 0], det_[:, 0])
        yy1 = np.maximum(det_[0, 1], det_[:, 1])
        xx2 = np.minimum(det_[0, 2], det_[:, 2])
        yy2 = np.minimum(det_[0, 3], det_[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o_ = inter / (area_[0] + area_[:] - inter)

        merge_index_ = np.where(o_ >= 0.3)[0]
        det_accu_ = det_[merge_index_, :]
        det_ = np.delete(det_, merge_index_, 0)

        if merge_index_.shape[0] <= 1:
            continue
        det_accu_[:, 0:4] = det_accu_[:, 0:4] * np.tile(det_accu_[:, -1:], (1, 4))
        max_score_ = np.max(det_accu_[:, 4])
        det_accu_sum_ = np.zeros((1, 5))
        det_accu_sum_[:, 0:4] = np.sum(det_accu_[:, 0:4], axis=0) / np.sum(det_accu_[:, -1:])
        det_accu_sum_[:, 4] = max_score_
        try:
            dets_ = np.row_stack((dets_, det_accu_sum_))
        except:
            dets_ = det_accu_sum_

    dets_ = dets_[0:750, :]
    return dets_


def load_models():
    print('build network')
    net = build_net('test', num_classes=2, model='dark')
    net.eval()
    # �޸�1������ѵ���õ�Ȩ��·��
    # �޸�2��map_location='cpu' ����࿨�����Ȩ�ؼ����豸��ͻ
    net.load_state_dict(torch.load('weights/dark/dsfd.pth', map_location='cpu'))

    if use_cuda:
        net = net.cuda()
        # priors_pal1/pal2 在 DAINet.__init__ 里创建为 CPU tensor，
        # 不随 net.cuda() 自动搬移，需手动移到 GPU
        if hasattr(net, 'priors_pal1'):
            net.priors_pal1 = net.priors_pal1.cuda()
        if hasattr(net, 'priors_pal2'):
            net.priors_pal2 = net.priors_pal2.cuda()

    return net


if __name__ == '__main__':

    ''' Parameters '''
    USE_MULTI_SCALE = True
    MY_SHRINK = 1

    # USE_MULTI_SCALE = False
    # MY_SHRINK = 2

    # �������Ŀ¼
    save_path = './result/'

    def load_images():
        # �޸�3���������ͼƬ·����֧�� 0.png ~ 99.png��
        imglist = sorted(
            glob.glob('dataset/track1.2_test_sample/track1.2_test_sample/*.png'),
            key=lambda p: int(Path(p).stem)   # ������˳�����У�0��99
        )
        return imglist

    ''' Main Test '''
    net = load_models()
    img_list = load_images()

    if not img_list:
        raise FileNotFoundError('No images found. Check path: dataset/track1.2_test_sample/track1.2_test_sample/*.png')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f'���ҵ� {len(img_list)} ��ͼƬ����ʼ���...')

    for now, img_path in enumerate(img_list):
        print('Processing: {}/{} �� {}'.format(now + 1, len(img_list), img_path))

        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = np.array(image)

        max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5
        max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink

        if USE_MULTI_SCALE:
            with torch.no_grad():
                det0 = detect_face(image, MY_SHRINK)
                det1 = flip_test(image, MY_SHRINK)
                [det2, det3] = multi_scale_test(image, max_im_shrink)
                det4 = multi_scale_test_pyramid(image, max_im_shrink)
            det = np.row_stack((det0, det1, det2, det3, det4))
            dets = bbox_vote(det)
        else:
            with torch.no_grad():
                dets = detect_face(image, MY_SHRINK)

        # ���д�� result/0.txt ~ result/99.txt
        out_file = os.path.join(save_path, Path(os.path.basename(img_path)).stem + '.txt')
        with open(out_file, 'w') as fout:
            for i in range(dets.shape[0]):
                xmin, ymin, xmax, ymax, score = dets[i]
                fout.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.6f}\n'.format(
                    xmin, ymin, xmax, ymax, score))

    print('��ɣ���������� {} Ŀ¼�£��� {} �� .txt �ļ�'.format(save_path, len(img_list)))