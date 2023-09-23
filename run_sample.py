# Original Code: https://github.com/jiwoon-ahn/irn

import argparse
import os
import numpy as np
import torch
import random
from misc import pyutils

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & (2 ** 32 - 1))
    random.seed(seed & (2 ** 32 - 1))
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", default='exp_fpr', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cam_weight_path", default='sess/res50_cam_orig.pth', type=str)
    parser.add_argument("--tempt", default=13, type=float)
    parser.add_argument("--num_cluster", default=10, type=int)
    parser.add_argument("--mask_thresh", default=0.1, type=float)
    parser.add_argument("--RC_weight", default=12e-2, type=float)
    parser.add_argument("--PR_weight", default=15e-5, type=float)
    parser.add_argument("--adv_iter", default=19, type=int) # reduce from 27 to 19 to save time

    # Environment
    parser.add_argument("--num_workers", default=16, type=int)  # os.cpu_count()//2
    parser.add_argument("--voc12_root", default='Dataset/VOC2012_SEG_AUG/', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thresh", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.25, type=float)
    parser.add_argument("--conf_bg_thres", default=0.10, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25, type=float)
    parser.add_argument("--sem_seg_bg_thres", default=0.20, type=float)
    parser.add_argument("--np_power", default=1.5, type=float)

    # Output Path
    parser.add_argument("--log_name", default="fpr.log", type=str)
    parser.add_argument("--cam_weights_name", default="sess/res50_fpr.pth", type=str)
    parser.add_argument("--irn_weights_name", default="sess/res50_fpr_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="result/fpr", type=str)
    parser.add_argument("--cam_vis_out_dir", default="result/fpr_vis", type=str)
    parser.add_argument("--advcam_out_dir", default="result/fpr_advcam", type=str)
    parser.add_argument("--ir_label_out_dir", default="result/ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="result/sem_seg", type=str)
    parser.add_argument("--ins_seg_out_dir", default="result/ins_seg", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=True, type=bool)
    parser.add_argument("--make_cam_pass", default=True, type=bool)
    parser.add_argument("--eval_cam_pass", default=True, type=bool)
    parser.add_argument("--adv_cam_pass", default=True, type=bool)
    parser.add_argument("--cam_to_ir_label_pass", default=True, type=bool)
    parser.add_argument("--train_irn_pass", default=True, type=bool)
    parser.add_argument("--make_ins_seg_pass", default=False, type=bool)
    parser.add_argument("--eval_ins_seg_pass", default=False, type=bool)
    parser.add_argument("--make_sem_seg_pass", default=True, type=bool)
    parser.add_argument("--eval_sem_seg_pass", default=True, type=bool)

    args = parser.parse_args()
    setup_seed(args.seed)
    args.cam_out_dir = os.path.join(args.exp_name, args.cam_out_dir)
    args.cam_vis_out_dir = os.path.join(args.exp_name, args.cam_vis_out_dir)
    args.advcam_out_dir = os.path.join(args.exp_name, args.advcam_out_dir)
    args.ir_label_out_dir = os.path.join(args.exp_name, args.ir_label_out_dir)
    args.sem_seg_out_dir = os.path.join(args.exp_name, args.sem_seg_out_dir)
    args.cam_weights_name = os.path.join(args.exp_name, args.cam_weights_name)
    args.irn_weights_name = os.path.join(args.exp_name, args.irn_weights_name)

    os.makedirs(args.exp_name+'/sess', exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.cam_vis_out_dir, exist_ok=True)
    os.makedirs(args.advcam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)

    pyutils.Logger(os.path.join(args.exp_name, args.log_name))
    print(vars(args))
    if args.train_cam_pass is True:
        import step.train_fpr
        step.train_fpr.run(args)

    if args.make_cam_pass is True:
        import step.make_cam
        # args.train_list = args.train_list.replace('train_aug','train')
        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        final_miou = []
        for i in range(3, 15):
            t = i / 100.0
            args.cam_eval_thres = t
            miou, precision, recall = step.eval_cam.run(args)
            final_miou.append(miou)
        print(args.cam_out_dir)
        print(final_miou)
        print(np.max(np.array(final_miou)))

    # print('eval crf')
    # import step.eval_crf
    # step.eval_crf.run(args)

    if args.adv_cam_pass is True:
        from get_advcam import adv_cam
        timer = pyutils.Timer('step.adv_cam:')
        adv_cam(args.advcam_out_dir, args.cam_weights_name, args.adv_iter)

    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels

        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)

