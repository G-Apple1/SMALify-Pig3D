from tqdm import tqdm
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from metrics import Metrics
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from smal_fitter import SMALFitter
import pickle as pkl
import torch
import imageio
from data_loader import load_badja_sequence, load_stanford_sequence
import trimesh
from tqdm import trange

import os, time
import config

class ImageExporter():
    def __init__(self, output_dir, filenames):
        self.output_dirs = self.generate_output_folders(output_dir, filenames)
        self.stage_id = 0
        self.epoch_name = 0

    def generate_output_folders(self, root_directory, filename_batch):
        if not os.path.exists(root_directory):
            os.mkdir(root_directory)

        output_dirs = [] 
        for filename in filename_batch:
            filename_path = os.path.join(root_directory, os.path.splitext(filename)[0])
            output_dirs.append(filename_path)
            if not os.path.exists(filename_path):
                os.mkdir(filename_path)
        
        return output_dirs

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces):
        imageio.imsave(os.path.join(self.output_dirs[global_id], "st{0}_ep{1}.png".format(self.stage_id, self.epoch_name)), collage_np)

        # Export mesh
        vertices = vertices[batch_id].cpu().numpy()
        mesh = trimesh.Trimesh(vertices = vertices, faces = faces, process = False)
        mesh.export(os.path.join(self.output_dirs[global_id], "st{0}_ep{1}.ply".format(self.stage_id, self.epoch_name)))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config.PIG_PATH = "../"

    dataset = "pig_data"

    test_list = ['000994', '000906', '000635', '000352', '000372']##
    len_data = len(test_list)
    pbar = tqdm(total=len_data, desc="imgs")

    for id_0, n0 in enumerate(test_list):
        print(n0,"====================")
        pbar.update(1)
        name = n0
        if dataset == "badja":
            data, filenames = load_badja_sequence(
                config.BADJA_PATH, name,
                config.CROP_SIZE, image_range=config.IMAGE_RANGE)
        elif dataset == "pig_data":
            data, filenames = load_stanford_sequence(
                config.PIG_PATH, name,
                config.CROP_SIZE)
        else:
            data, filenames = load_stanford_sequence(
                config.STANFORD_EXTRA_PATH, name,
                config.CROP_SIZE
            )

        dataset_size = len(filenames)
        print ("Dataset size: {0}".format(dataset_size))

        assert config.SHAPE_FAMILY >= -1, "Shape family should be greater than -1"

        use_unity_prior = config.SHAPE_FAMILY == 1 and not config.FORCE_SMAL_PRIOR

        if not use_unity_prior and not config.ALLOW_LIMB_SCALING:
            print("WARNING: Limb scaling is only recommended for the new Unity prior. TODO: add a regularizer to constrain scale parameters.")
            config.ALLOW_LIMB_SCALING = False

        image_exporter = ImageExporter(config.OUTPUT_DIR, filenames)

        model = SMALFitter(device, data, config.WINDOW_SIZE, config.SHAPE_FAMILY, use_unity_prior)
        for stage_id, weights in enumerate(np.array(config.OPT_WEIGHTS).T):
            opt_weight = weights[:6]
            w_temp = weights[6]
            epochs = int(weights[7])
            lr = weights[8]

            '''
            shape(beta): 8
            log_beta_scales: (1,6)
            global_rotation: (1,3)
            trans: (1,3)
            joint_rotations(theta): (1,34,3)
            '''
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

            if stage_id == 0:
                model.joint_rotations.requires_grad = False
                model.betas.requires_grad = False
                model.log_beta_scales.requires_grad = False
                target_visibility = model.target_visibility.clone()
                model.target_visibility *= 0
                # Turn on only torso points
                model.target_visibility[:, config.TORSO_JOINTS] = target_visibility[:, config.TORSO_JOINTS]
            else:
                model.joint_rotations.requires_grad = True
                model.betas.requires_grad = True
                if config.ALLOW_LIMB_SCALING:
                    model.log_beta_scales.requires_grad = True
                model.target_visibility = data[-1].clone()

            # t = trange(epochs, leave=False)#leave=True 参数表示在循环完成后保留进度条。
            # for epoch_id in t:
            for epoch_id in range(epochs):
                image_exporter.stage_id = stage_id
                image_exporter.epoch_name = str(epoch_id)

                acc_loss = 0
                optimizer.zero_grad()
                for j in range(0, dataset_size, config.WINDOW_SIZE):
                    batch_range = list(range(j, min(dataset_size, j + config.WINDOW_SIZE)))
                    loss, losses = model(batch_range, opt_weight, stage_id)
                    acc_loss += loss.mean()

                joint_loss, global_loss, trans_loss = model.get_temporal(w_temp)

                acc_loss = acc_loss + joint_loss + global_loss + trans_loss
                acc_loss.backward()
                optimizer.step()

                if epoch_id % config.VIS_FREQUENCY == 0:
                    try:
                        _, _, _, _ = model.generate_visualization(image_exporter)
                    except:
                        print("Fitting failed: ", n0)
                        continue


        image_exporter.stage_id = 10
        image_exporter.epoch_name = str(0)
        try:
            # Final stage
            synth_silhouettes, gtseg, pred_keypoints, gt_keypoints = model.generate_visualization(image_exporter)
            gtseg = torch.div(gtseg[0], 255, rounding_mode='trunc')
        except:
            print("Fitting failed: ", n0)
            continue

        '''evaluation code from BARC'''
        preds = {}
        pck_thresh = 0.15
        has_seg = torch.tensor([True], device=device)
        KEYPOINT_GROUPS = {'ears': [14, 15, 18, 19], 'face': [16, 17], 'legs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                           'tail': [12, 13]}
        EVAL_KEYPOINTS = [
            0, 1, 2,  # left front
            3, 4, 5,  # left rear
            6, 7, 8,  # right front
            9, 10, 11,  # right rear
            12, 13,  # tail start -> end
            14, 15,  # left ear, right ear
            16, 17,  # nose, chin
            18, 19]  # left ear tip, right ear tip
        if id_0 == 0:
            pck = np.zeros((len_data))
            pck_by_part = {group: np.zeros((len_data)) for group in KEYPOINT_GROUPS}
            acc_sil_2d = np.zeros(len_data)
        img_border_mask = torch.all(gtseg > 1.0 / 256, dim=0).unsqueeze(0).float()

        preds['acc_PCK'] = Metrics.PCK(pred_keypoints, gt_keypoints.cuda(), gtseg, has_seg, idxs=EVAL_KEYPOINTS, thresh_range=[pck_thresh])
        preds['acc_IOU'] = Metrics.IOU(synth_silhouettes, gtseg, img_border_mask, mask=has_seg)

        for group, group_kps in KEYPOINT_GROUPS.items():
            preds[f'{group}_PCK'] = Metrics.PCK(pred_keypoints, gt_keypoints.cuda(), gtseg, has_seg, thresh_range=[pck_thresh], idxs=group_kps)

        # add results for all images in this batch to lists
        pck[id_0] = preds['acc_PCK'].data.cpu().numpy()
        acc_sil_2d[id_0] = preds['acc_IOU'].data.cpu().numpy()
        for part in pck_by_part:
            pck_by_part[part][id_0] = preds[f'{part}_PCK'].data.cpu().numpy()
    pbar.close()
    iou = np.nanmean(acc_sil_2d)
    pck = np.nanmean(pck)
    pck_legs = np.nanmean(pck_by_part['legs'])
    pck_tail = np.nanmean(pck_by_part['tail'])
    pck_ears = np.nanmean(pck_by_part['ears'])
    pck_face = np.nanmean(pck_by_part['face'])
    print('------------------------------------------------')
    print("iou:         {:.2f}".format(iou*100))
    print('                                                ')
    print("pck:         {:.2f}".format(pck*100))
    print('                                                ')
    print("pck_legs:    {:.2f}".format(pck_legs*100))
    print("pck_tail:    {:.2f}".format(pck_tail*100))
    print("pck_ears:    {:.2f}".format(pck_ears*100))
    print("pck_face:    {:.2f}".format(pck_face*100))
    print('------------------------------------------------')


if __name__ == '__main__':
    main()
