# Original Code: https://github.com/jiwoon-ahn/irn

import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import importlib
import numpy as np
import os
import cv2

import voc12.dataloader
from step.train_utils import validate
from chainercv.evaluations import calc_semantic_segmentation_confusion
from PIL import Image
from misc import pyutils, torchutils, imutils
from chainercv.datasets import VOCSemanticSegmentationDataset
from step.cluster_utils import get_regional_cluster


def get_region_contrast_loss(model, data, target, clustered_pos, clustered_neg, temperature, mask_thresh):
    """
    calculating the distance between the image feature and clustered embedding
    Args:
        model:
        data: imgs with shape [bs, 3, h, w]
        target: with shape [bs, num_cls]
        clustered_pos: the positive regional feature embedding. [num_cls, num_pos_k, C]
        clustered_neg: the negative regional feature embedding. [num_cls, num_neg_k, C]
        temperature: the hyper-param temperature
        mask_thresh: the value to filter background
    Return:
        loss_pos: the distances with shape [num_present_cls, 1] the 1 is argmin of $num_k distance
        loss_neg: the distances with shape [num_present_cls, num_neg_k]
        logits: the logits for each class [bs, num_cls]
    """

    def exp_similarity(a, b, temperature):
        """
        calculate the distance of a, b. return the exp{-L1(a,b)}
        """
        if len(b) == 0 or len(a) == 0:  # no clusters
            return torch.Tensor([0.5]).to(a.device)

        dis = ((a - b) ** 2 + 1e-4).mean(1)
        dis = torch.sqrt(dis)
        dis = dis / temperature + 0.1  # prevent too large gradient
        return torch.exp(-dis)

    pos_loss = []
    neg_loss = []

    data, target = data.cuda(), target.cuda()
    data.requires_grad_()
    _, pred_features, cams, logits = model.forward_feat(data)

    norm_cams = F.relu(cams) / (F.adaptive_max_pool2d(cams, 1) + 1e-4)
    norm_cams = (norm_cams > mask_thresh).float().detach().flatten(2)
    out_features = pred_features.flatten(2)

    for logit, feat, norm_cam, cam, gt in zip(logits, out_features, norm_cams, cams, target):
        for idx, is_exist in enumerate(gt):

            if cam[idx].max() <= 0 or len(clustered_pos[idx]) == 0 or len(clustered_neg[idx]) == 0:
                continue

            cls_norm_cam = norm_cam[idx]
            region_feat = (feat * cls_norm_cam[None]).sum(1) / (cls_norm_cam.sum() + 1e-5)

            if is_exist:
                # distance to pos clusters
                pos_feat = clustered_pos[idx].mean(0, keepdim=True)
                pos_prob = exp_similarity(region_feat[None], pos_feat, temperature=temperature)
                loss_pos = -torch.log(pos_prob)
                pos_loss.append(loss_pos.squeeze())

                # distance to neg clusters
                neg_feat = clustered_neg[idx]
                neg_prob = exp_similarity(region_feat[None], neg_feat, temperature=temperature).max()
                loss_neg = -torch.log(1 - neg_prob)
                neg_loss.append(loss_neg)

    loss_pos = torch.stack(pos_loss).mean() if len(pos_loss) > 0 else 0
    loss_neg = torch.stack(neg_loss).mean() if len(neg_loss) > 0 else 0
    return loss_pos, loss_neg,  logits, cams, pred_features


def get_pixel_rectification_loss(cams, label, pred_feats, pos_feats, neg_feats, mask_thresh):
    """
    Args:
        cams: with shape of [bs, 20, H ,W]
        label: with shape of [bs, 20]
        pred_feats: with shape of [bs, C, H, W]
        pos_feats: with shape of [20, N_cluster, C]
        neg_feats: with shape of [20, N_cluster, C]
        mask_thresh: the threshold to filter background
    Return:
        loss_revise: the loss for decreasing the prob of foreground pixels which is more closed to negative pixel
        loss_seg: the loss for supervising the segmentation output
    """

    def closest_dis(a, b):
        """
        Args:
            a: with shape of [1, C, HW]
            b: with shape of [num_clusters, C, 1]
        Return:
            dis: with shape of [HW]
        """
        if len(b) == 0 or len(a) == 0:  # no clusters
            return torch.Tensor([123456]).to(a.device)
        dis = ((a - b) ** 2).mean(1)
        return dis.min(0)[0]

    # 1. norm
    bs, num_cls = label.shape
    norm_cams = F.relu(cams) / ((F.adaptive_max_pool2d(cams, 1) + 1e-3).detach())
    norm_cams = norm_cams.flatten(2)[:, :-1]

    # 2. calculate the negative pixels
    pred_feats = pred_feats.flatten(2)
    cams = cams.flatten(2)
    probs = []  # the pixels to decrease
    for bs_id in range(bs):
        for cls_id in range(num_cls):
            if label[bs_id][cls_id]:
                dis_pos = closest_dis(pred_feats[bs_id][None], pos_feats[cls_id][..., None])
                dis_neg = closest_dis(pred_feats[bs_id][None], neg_feats[cls_id][..., None])

                fp_location = (norm_cams[bs_id][cls_id] > mask_thresh) * (dis_pos > dis_neg)
                fp_pixels = cams[bs_id][cls_id][fp_location]
                if len(fp_pixels) > 0:
                    probs.append(fp_pixels)


    # loss_PC
    if len(probs) > 0:
        probs = torch.cat(probs)
        loss_revise = probs.mean()
        # pixel_gt = torch.zeros_like(probs).to(probs.device)
        # loss_revise = F.binary_cross_entropy_with_logits(probs, pixel_gt).mean()
    else:
        loss_revise = 0
    return loss_revise


def process(id, args, thresh):
    label = np.array(Image.open(os.path.join(args.voc12_root, 'SegmentationClass', '%s.png' % id)))
    # print(os.path.join(args.cam_out_dir, id + '.npy'))
    cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
    if not ('high_res' in cam_dict):
        return np.zeros_like(label), label

    cams = cam_dict['high_res']
    cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=thresh)
    keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
    cls_labels = np.argmax(cams, axis=0)
    cls_labels = keys[cls_labels]
    return cls_labels.copy(), label


def eval_cam_sub(args):
    miou_best = 0
    eval_dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    preds = []
    labels = []
    # n_images = 0
    for thresh in [0.06, 0.08, 0.10, 0.12]:
        for i, id in enumerate(eval_dataset.ids):
            # n_images += 1
            cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
            cams = cam_dict['high_res']

            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=thresh)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            preds.append(cls_labels.copy())
            labels.append(eval_dataset.get_example_by_keys(i, (1,))[0])

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        confusion_np = np.array(confusion)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        miou = np.nanmean(gtjresj / denominator)
        if miou > miou_best:
            miou_best = miou

    return miou_best


def sub_cam_eval(args, model):
    model_cam = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model_cam.load_state_dict(model.state_dict(), strict=True)
    model_cam.eval()
    model_cam.cuda()
    dataset_cam = voc12.dataloader.VOC12ClassificationDatasetMSF("voc12/train.txt",
                                                                 voc12_root=args.voc12_root,
                                                                 scales=[0.5, 1.0, 1.5, 2.0])
    dataset_cam = torchutils.split_dataset(dataset_cam, 1)

    databin = dataset_cam[0]
    data_loader_cam = DataLoader(databin, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    with torch.no_grad():
        for iter, pack in enumerate(data_loader_cam):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            outputs = [model_cam.forward_ms_flip(img[0].cuda(non_blocking=True)) for img in
                       pack['img']]  # b x 20 x w x h
            strided_cam = [F.interpolate(o[None], strided_size, mode='bilinear', align_corners=False)[0] for o in
                           outputs]
            strided_cam = torch.sum(torch.stack(strided_cam), 0)
            highres_cam = [F.interpolate(o[:, None], strided_up_size, mode='bilinear', align_corners=False) for o in
                           outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0]

            if len(valid_cat) == 0:
                np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                        {"keys": valid_cat})
                continue
            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5
            strided_cam = strided_cam.cpu().numpy()

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            highres_cam = highres_cam.cpu().numpy()

            # save cams
            # for cls_id, cam in zip(valid_cat, highres_cam):
            #     orig_img = cv2.imread('Dataset/VOC2012_SEG_AUG/JPEGImages/{}.jpg'.format(img_name))
            #     paint_cam = orig_img * 0.3 + np.expand_dims(cam*255, -1) * 0.7
            #     cv2.imwrite(os.path.join(args.cam_out_dir, '{}_{}.png'.format(img_name, int(cls_id))), paint_cam)

            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam, "high_res": highres_cam})

    miou = eval_cam_sub(args)
    return miou


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    pth_path = args.cam_weight_path
    model.load_state_dict(torch.load(pth_path), strict=False)
    model = model.cuda()

    max_step = (len(np.loadtxt(args.train_list, dtype=np.int32)) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()
    best_sub_miou = 0
    for ep in range(args.cam_num_epoches):
        print('Epoch %d/%d' % (ep + 1, args.cam_num_epoches))

        # 1. online cluster
        print('generating positive clusters and negative clusters ...')
        vis_path = os.path.join(args.exp_name, '{}_imgs_for_cluster'.format(ep))
        pos_feats, neg_feats, cur_neg_feats_unshared = get_regional_cluster(vis_path, model, num_cluster=args.num_cluster, region_thresh=args.mask_thresh)

        # 3. train
        train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                    resize_long=(320, 640), hor_flip=True,
                                                                    crop_size=512, crop_method="random")
        train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                       shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

        model.train()
        for step, pack in enumerate(train_data_loader):
            if ep == args.cam_num_epoches - 1:
                if (step > len(train_data_loader) * 0.75) and (step % 10 == 0):  # inherited from W-OoD
                    now_miou = sub_cam_eval(args, model)
                    if now_miou > best_sub_miou:
                        torch.save(model.state_dict(), args.cam_weights_name)
                        best_sub_miou = now_miou

            img, label = pack['img'], pack['label'].cuda(non_blocking=True)

            # ======================== loss_RC
            loss_dis_pos, loss_dis_neg, logits, cams, pred_feats = get_region_contrast_loss(model, img, label, pos_feats,
                                                                             neg_feats, temperature=args.tempt, mask_thresh=args.mask_thresh)
            loss_RC = (loss_dis_pos +loss_dis_neg)* args.RC_weight

            # ========================= loss_PR
            loss_PR = get_pixel_rectification_loss(cams, label, pred_feats, pos_feats, cur_neg_feats_unshared, args.mask_thresh)
            loss_PR = loss_PR * args.PR_weight

            # ========================= BCE loss
            loss_bce = F.binary_cross_entropy_with_logits(logits, label, reduction='mean')

            # ========================= focal BCE loss
            # loss_bce = - (label * F.logsigmoid(logits) + (F.sigmoid(logits)**args.gamma) * (1-label) * F.logsigmoid(-logits)).mean()

            loss = loss_bce + loss_RC + loss_PR
            avg_meter.add({'loss_bce': loss_bce.item(),
                           'loss_RC': loss_RC,
                           'loss_PR': loss_PR,
                           'loss': loss})

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(param_groups[0], max_norm=5)
            optimizer.step()

            if (optimizer.global_step - 1) % 20 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss_BCE:%.4f' % (avg_meter.pop('loss_bce')),
                      'loss_RC:%.4f' % (avg_meter.pop('loss_RC')),
                      'loss_PR:%.4f' % (avg_meter.pop('loss_PR')),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        # 4. save internal weight
        torch.save(model.state_dict(), args.cam_weights_name.replace('.pth', f'epoch{ep+1}.pth'))

        # 5. eval after each epoch
        validate(model, val_data_loader)
        timer.reset_stage()

    now_miou = sub_cam_eval(args, model)

    if now_miou > best_sub_miou:
        torch.save(model.state_dict(), args.cam_weights_name)

    torch.cuda.empty_cache()
