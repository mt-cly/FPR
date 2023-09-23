# Original Code: https://github.com/jiwoon-ahn/irn

import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import imageio

import voc12.dataloader
from misc import torchutils, indexing
from PIL import Image

cudnn.enabled = True
palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
         64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
         0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
         64,64,0,  192,64,0,  64,192,0, 192,192,0]

def _work(process_id, model, dataset, args):

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    print(len(data_loader))
    with torch.no_grad(), cuda.device(process_id % n_gpus):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            # if os.path.exists(os.path.join(args.sem_seg_out_dir+'_realth_%.2f' % (0.28), img_name + '.png')):
            #     print("passed")
            #     continue
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(args.advcam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cams = np.power(cam_dict['cam'], args.np_power)
            # for cam in cams:
            #     print(cam.shape, cam.max())
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()
            # print(cam_downsized_values.shape, edge.shape)
            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()
            rw_pred = keys[rw_pred]
            out = Image.fromarray(rw_pred.astype(np.uint8), mode='P')
            out.putpalette(palette)
            out.save(os.path.join(os.path.join(args.sem_seg_out_dir, img_name + '_palette.png')))
            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
            # heatmap
            # import cv2
            # heatmap = cv2.applyColorMap((rw_up.max(0)[0]*255).cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
            # orig_img = cv2.imread('Dataset/VOC2012_SEG_AUG/JPEGImages/{}.jpg'.format(img_name))
            # heatmap = orig_img * 0.5 + heatmap * 0.5
            # cv2.imwrite(os.path.join(args.sem_seg_out_dir, img_name + '_heatmap.png'), heatmap)


            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    print(args.irn_weights_name)
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()
    n_gpus = torch.cuda.device_count() * 3
    args.infer_list = "voc12/train.txt"
    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list,
                                                             voc12_root=args.voc12_root,
                                                             scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()
