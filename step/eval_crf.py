# Original Code: https://github.com/jiwoon-ahn/irn
#################################
# bg_thresh     miou
#    0.19       64.9
#    0.2        65.1
#    0.21       65.33
#    0.22       65.36
#    0.25       65.05
############################

import os
import numpy as np
import imageio

from torch import multiprocessing
from torch.utils.data import DataLoader

import voc12.dataloader
from misc import torchutils, imutils

from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

n_class = 21
def total_confusion_to_class_confusion(data):

    confusion_c = np.zeros((n_class, 2, 2))
    for i in range(n_class):
        confusion_c[i, 0, 0] = data[i, i]
        confusion_c[i, 0, 1] = np.sum(data[i, :]) - data[i, i]
        confusion_c[i, 1, 0] = np.sum(data[:, i]) - data[i, i]
        confusion_c[i, 1, 1] = np.sum(data) - np.sum(data[i, :]) - np.sum(data[:, i]) + data[i, i]
    return confusion_c

def work(infer_dataset, args, fg_thresh=0.25, bg_thresh=0.10):

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=0, pin_memory=False)
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    preds = []
    labels = []
    n_images=0
    for iter, pack in enumerate(infer_data_loader):
        if iter%50 == 0:
            print('iter:{}/{}'.format(iter, len(infer_data_loader)))
        n_images += 1
        img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
        # if os.path.exists(os.path.join(args.ir_label_out_dir, img_name + '.png')):
        #     continue
        img = pack['img'][0].numpy()
        # TODO
        cam_dict = np.load(os.path.join(args.advcam_out_dir, img_name + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

        # 1. find confident fg & bg
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=fg_thresh)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        fg_conf = keys[pred]

        bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=bg_thresh)
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 0
        conf[bg_conf + fg_conf == 0] = 0

        # cam_dict = np.load(os.path.join('/home/liyi/proj/w-ood/step/seam/resnet38_SEAM_FPR_tempe1_maskthresh0.2_wer1.0_wecr1.0_w10.3_w20.00015_numc10_epoch3', img_name + '.npy'), allow_pickle=True).item()
        # keys = np.array([0]+ [key+1 for key in cam_dict.keys()])
        # cams = np.array([cam_dict[key] for key in cam_dict.keys()])
        # cam_bg = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=fg_thresh)
        # cam_bg = np.argmax(cam_bg, axis=0)
        # pred = imutils.crf_inference_label(img, cam_bg, n_labels=keys.shape[0])
        # conf = keys[pred]

        preds.append(conf)
        labels.append(dataset.get_example_by_keys(iter, (1,))[0])

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    confusion_np = np.array(confusion)
    confusion_c = total_confusion_to_class_confusion(confusion_np).astype(float)
    precision, recall = [], []
    for i in range(n_class):
        recall.append(confusion_c[i, 0, 0] / np.sum(confusion_c[i, 0, :]))
        precision.append(confusion_c[i, 0, 0] / np.sum(confusion_c[i, :, 0]))

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    print("\n")

    print("fg_threshold:", fg_thresh, "bg_threshold:", bg_thresh, 'miou:', np.nanmean(iou), "i_imgs", n_images, "precision",
          np.mean(np.array(precision)), "recall", np.mean(np.array(recall)))
    print("\n")

    print(iou)
    print(precision)
    print(recall)

def run(args):
    dataset = voc12.dataloader.VOC12ImageDataset("voc12/train.txt", voc12_root=args.voc12_root, img_normal=None, to_torch=False)
    for fg_thresh in range(30,40,2):
        bg_thresh = fg_thresh
        work(dataset,args,  fg_thresh/100., bg_thresh/100.)
