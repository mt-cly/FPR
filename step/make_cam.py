# Original Code: https://github.com/jiwoon-ahn/irn


import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
import cv2
import numpy as np
import importlib
import os
from step.train_utils import show_cam_on_image
import voc12.dataloader
from misc import torchutils, imutils
from misc.imutils import show_cam_on_image

cudnn.enabled = True


def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id % n_gpus):

        model.cuda()
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model.forward_ms_flip(img[0].cuda(non_blocking=True), return_norelu=True) for img in pack['img']]
            # print(len(outputs), len(outputs[0]), outputs[0][0].shape, outputs[0][1].shape)
            logits = []
            for o in outputs:
                logits.append(torchutils.gap2d(o[1].unsqueeze(0)).squeeze(0))
            # logits = [torchutils.gap2d(o[1].unsqueeze(0)).squeeze(0) for o in outputs]
            # print(logits[0])
            outputs = [o[0] for o in outputs]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)
            valid_cat = torch.nonzero(label)[:, 0]

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]


            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5
            strided_cam = strided_cam.cpu()

            # for s_idx, s_cam in enumerate(strided_cam):
            #     c_now = valid_cat[s_idx]
            #     strided_cam[s_idx] = strided_cam[s_idx] * torch.sigmoid(logits[1][c_now]/2)
            # print(torch.sigmoid(logits[1][c_now]/2))
            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            highres_cam = highres_cam.cpu().numpy()

            # save attn map
            orig_img = cv2.imread(os.path.join(args.voc12_root, 'JPEGImages', img_name+'.jpg'))
            for i, cls_idx in enumerate(torch.nonzero(label)[:, 0]):
                show_cam_on_image(orig_img, highres_cam[i], os.path.join(args.cam_vis_out_dir, img_name+'_'+str(cls_idx)+'.png'))

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam, "high_res": highres_cam})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    # print(model)
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count() * 2

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list, voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()