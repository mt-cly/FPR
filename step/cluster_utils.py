import random

import torch
import torch.multiprocessing
from torch.multiprocessing import Manager
from torch.utils.data import DataLoader
import voc12.dataloader
import torch.nn.functional as F
import cv2
from sklearn.cluster import KMeans
import os
from step.train_utils import show_cam_on_image
from misc import torchutils
from misc import pyutils
import numpy as np

cls_names = ['0_aeroplane', '1_bicycle', '2_bird', '3_boat', '4_bottle', '5_bus', '6_car', '7_cat', '8_chair',
             '9_cow', '10_diningtable', '11_dog', '12_horse', '13_motorbike', '14_person', '15_pottedplant',
             '16_sheep', '17_sofa', '18_train', '19_tvmonitor']

super_class = ['PERSON', 'ANIMAL', 'VEHICLE', 'INDOOR']
num_sub_class = {super_class[0]: 1, super_class[1]: 6, super_class[2]: 7, super_class[3]: 6}
super_class_map = {0: super_class[2], 1: super_class[2], 2: super_class[1], 3: super_class[2], 4: super_class[3],
                   5: super_class[2], 6: super_class[2], 7: super_class[1], 8: super_class[3], 9: super_class[1],
                   10: super_class[3], 11: super_class[1], 12: super_class[1], 13: super_class[2], 14: super_class[0],
                   15: super_class[3], 16: super_class[1], 17: super_class[3], 18: super_class[2], 19: super_class[3]}


def without_shared_feats(cls_idx, tgt):
    num_cls = len(tgt)
    # TODO
    return super_class_map[cls_idx] not in [super_class_map[i] for i in range(num_cls) if tgt[i]] and num_sub_class[super_class_map[cls_idx]]>1


def list2tensor(feature_list):
    if len(feature_list) > 0:
        return torch.stack(feature_list)
    else:
        return torch.Tensor([])

# TODO SS/MS
def _get_cluster(model, num_cls, region_thresh, vis_path, num_cluster, single_stage=False, mask_img=False, sampling_ratio=1.):
    trainmsf_dataset = voc12.dataloader.VOC12ClassificationDatasetMSF("voc12/train_aug.txt",
                                                                      voc12_root="Dataset/VOC2012_SEG_AUG/",
                                                                      scales=(1.0, 0.5, 1.5, 2.0))
    trainmsf_dataloader = DataLoader(trainmsf_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                                     drop_last=True)

    pos_clusters = [[] for _ in range(num_cls)]
    neg_info = [[] for _ in range(num_cls)]
    neg_clusters = [[] for _ in range(num_cls)]
    neg_clusters_unshared = [[] for _ in range(num_cls)]

    with torch.no_grad():
        model = model.cuda()
        model.eval()
        # 1. collect feature embeddings
        for img_idx, pack in enumerate(trainmsf_dataloader):
            #if random.random() > sampling_ratio:
            #    continue

            if img_idx % 1000 == 0:
                print('processing regions features in img of {}/{}'.format(img_idx, len(trainmsf_dataloader)))
            ms_imgs = pack['img']
            tgt = pack['label'][0].cuda()
            name = pack['name'][0]

            # 1-1. ss/ms inferring
            cams_list, feats_list = [], []
            for idx, img in enumerate(ms_imgs):
                _, feat, cam, _ = model.forward_feat(img[0].cuda(non_blocking=True))
                cam = cam[:, :num_cls]
                if idx == 0:
                    size = cam.shape[-2:]
                else:
                    feat = F.interpolate(feat, size, mode='bilinear', align_corners=True)
                    cam = F.interpolate(cam, size, mode='bilinear', align_corners=True)
                feat = (feat[0] + feat[1].flip(-1)) / 2
                cam = (cam[0] + cam[1].flip(-1)) / 2
                cams_list.append(cam)
                feats_list.append(feat)
                if single_stage:
                    break
            ms_cam = torch.stack(cams_list).mean(0)  # [num_cls, H, W]
            ms_feat = torch.stack(feats_list).mean(0)  # [C, H, W]
            _, h, w = ms_cam.shape

            # 1-2. normalize
            pred_prob = ms_cam.flatten(1).mean(1)
            norm_ms_cam = F.relu(ms_cam) / (F.adaptive_max_pool2d(F.relu(ms_cam), (1, 1)) + 1e-5)

            # 1-3. collect regional features
            orig_img = cv2.imread('Dataset/VOC2012_SEG_AUG/JPEGImages/{}.jpg'.format(name))
            for cls_idx, is_exist in enumerate(tgt[:num_cls]):
                if is_exist:
                    region_feat = (ms_feat[:, norm_ms_cam[cls_idx] > region_thresh])
                    if region_feat.shape[-1] > 0:
                        pos_clusters[cls_idx].append(region_feat.mean(1))
                        # # vis
                        # if cls_idx == 18:
                        #     if_activate = (norm_ms_cam[cls_idx] > region_thresh).reshape(h, w)[None, None].float()
                        #     if_activate = F.interpolate(if_activate, orig_img.shape[:2], mode='nearest')
                        #     if_activate = if_activate.squeeze().cpu().numpy()
                        #     if_activate = orig_img * 0.5 + if_activate[..., None] * 255 * 0.5
                        #     cv2.imwrite('{}/{}_pos_{}.png'.format(vis_path, cls_idx, name), if_activate)
                        #     # vis the pixel-level rectification
                        #     def closest_dis(a, b):
                        #         return ((a[None]-b[...,None,None])**2).mean(1).min(0)[0]
                        #     cluster = torch.load('/home/liyi/proj/w-ood/cluster10_disloss0.10_temper13_revise5e-5/0_imgs_for_cluster/clusters.pth')
                        #     pos_cluster = cluster['pos'][18]
                        #     neg_cluster = cluster['neg_unshared'][18]
                        #     if_closed = closest_dis(ms_feat, pos_cluster) > closest_dis(ms_feat, neg_cluster)
                        #     if_closed = (if_closed * (norm_ms_cam[18] > region_thresh)).float()
                        #     if_closed = F.interpolate(if_closed[None,None], orig_img.shape[:2], mode='nearest')
                        #     if_closed = if_closed.squeeze().cpu().numpy()
                        #     red_cloed = orig_img * 0.5
                        #     red_cloed[..., 2] = red_cloed[..., 2] + if_closed * 250 * 0.5
                        #     cv2.imwrite('{}/{}_pos_{}_rectification.png'.format(vis_path, cls_idx, name), red_cloed)

                if not is_exist:
                    cam_mask = norm_ms_cam[cls_idx] > region_thresh
                    if cam_mask.sum() > 0:
                        info = [pred_prob[cls_idx], img_idx, cam_mask, tgt[:num_cls]]
                        neg_info[cls_idx].append(info)
                        # if True and cls_idx == 18:
                        #     # # vis
                        #     vis_high_cam = F.interpolate(F.relu(norm_ms_cam[cls_idx][None,None]), orig_img.shape[:2], mode='bilinear', align_corners=False)
                        #     vis_high_cam = vis_high_cam.squeeze().cpu()
                        #     show_cam_on_image(orig_img, vis_high_cam, '{}/{}_neg_{}.png'.format(vis_path, cls_idx, name))

        # 2. get clusters from collected features
        for cls_idx in range(num_cls):
            # 2-1. positive cluster
            if len(pos_clusters[cls_idx]) > 0:
                pos_feats_np = torch.stack(pos_clusters[cls_idx]).cpu().numpy()
                num_k = min(num_cluster, len(pos_feats_np))
                centers = KMeans(n_clusters=num_k, random_state=0, max_iter=10).fit(pos_feats_np).cluster_centers_
                pos_clusters[cls_idx] = torch.from_numpy(centers).cuda()
            else:
                pos_clusters[cls_idx] = torch.Tensor([]).cuda()

            # 2-2. negative cluster
            probs = torch.stack([item[0] for item in neg_info[cls_idx]])
            num_k = min(num_cluster, len(neg_info[cls_idx]))
            top_prob_idx = torch.topk(probs, num_k)[1]
            for item_idx in top_prob_idx:
                prob, img_idx, cam_mask, tgt = neg_info[cls_idx][item_idx]
                pack = trainmsf_dataset.__getitem__(img_idx)
                ms_imgs, name = pack['img'], pack['name']
                feats_list = []
                for idx, img in enumerate(ms_imgs):
                    img_mask = F.interpolate(cam_mask[None, None].float(), img.shape[-2:], mode='nearest')
                    img_mask_flip = torch.cat([img_mask, img_mask.flip(-1)])
                    masked_img = torch.from_numpy(img.copy()).cuda() * img_mask_flip
                    if idx == 0:
                        alpha = 0.6
                        orig_img = cv2.imread('Dataset/VOC2012_SEG_AUG/JPEGImages/{}.jpg'.format(pack['name']))
                        cv2.imwrite('{}/{}_{}_orig.png'.format(vis_path, cls_names[cls_idx], name), orig_img)
                        fp_masked_img = img_mask[0].permute(1, 2, 0).cpu().numpy() * 255 * alpha + orig_img* (1-alpha)
                        cv2.imwrite('{}/{}_{}_prob{:.2f}.png'.format(vis_path, cls_names[cls_idx], name, prob),
                                    fp_masked_img)

                    if mask_img:
                        feat = model.forward_feat(masked_img)[1]
                    else:
                        unmask_img = torch.from_numpy(img.copy()).cuda()
                        feat = model.forward_feat(unmask_img)[1]

                    feat = F.interpolate(feat, cam_mask.shape, mode='bilinear', align_corners=True)
                    feat = (feat[0] + feat[1].flip(-1)) / 2
                    feats_list.append(feat)
                    if single_stage:
                        break
                ms_feat = torch.stack(feats_list).mean(0)  # [C, H, W]
                ms_feat = ms_feat[:, cam_mask]
                neg_clusters[cls_idx].append(ms_feat.mean(1))
                if without_shared_feats(cls_idx, tgt):
                    neg_clusters_unshared[cls_idx].append(ms_feat.mean(1))

            neg_clusters[cls_idx] = list2tensor(neg_clusters[cls_idx]).cuda()
            neg_clusters_unshared[cls_idx] = list2tensor(neg_clusters_unshared[cls_idx])

    return pos_clusters, neg_clusters, neg_clusters_unshared


def get_regional_cluster(vis_path, model, num_cls=20, num_cluster=10, region_thresh=0.1,sampling_ratio=1.):
    """
    Args:
        model: the training model
        num_cls: the number of classes
        num_pos_k: the number of positive cluster for each class
        num_neg_k: the number of negative cluster for each class
        region_thresh: the threshold for getting reliable regions
    Return:
        clustered_fp_feats: a tensor with shape [num_cls, num_k, C]
    """

    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    pos_clusters, neg_clusters, neg_clusters_unshared = _get_cluster(model, num_cls, region_thresh, vis_path,
                                                                     num_cluster, sampling_ratio=sampling_ratio)

    torch.save({'pos': pos_clusters, 'neg': neg_clusters, 'neg_unshared': neg_clusters_unshared},
               '{}/clusters.pth'.format(vis_path))

    return pos_clusters, neg_clusters, neg_clusters_unshared
