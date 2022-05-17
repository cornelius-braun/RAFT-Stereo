from __future__ import print_function, division
import sys
import os
from PIL import Image
sys.path.append('core')

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from raft_stereo import RAFTStereo, autocast
import stereo_datasets as datasets
from utils.utils import InputPadder
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_things(model, iters=32, mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation FlyingThings: %f, %f" % (epe, d1))
    return {'things-epe': epe, 'things-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)

        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}
    
    
@torch.no_grad()
def validate_endovis(model, iters=32, mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    aug_params = {}
    status = 'val'
    split='Endovis_split_RAFT/val_files.txt'
    val_dataset = datasets.Endovis(status, aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation Endovis: %f, %f" % (epe, d1))
    return {'endovis-epe': epe, 'endovis-d1': d1}
    
def load_image(imfile):
    transform_resize = transforms.Resize((256, 320))
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()
    
def get_bf(dataset_id):
    if dataset_id == 9:
        b = 4.132568439900515
        f = 1110.2546425040427
    elif dataset_id == 8:
        b = 4.348277165573036
        f = 1112.709358269966
    return b, f
    
def compute_errors(gt, pred, MIN_DEPTH=25, MAX_DEPTH=300):
    """Computation of error metrics between predicted and ground truth depths
    """
    mask = np.logical_and(gt >= MIN_DEPTH,  gt <= MAX_DEPTH)
    gt = gt[mask]
    pred = pred[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

@torch.no_grad()
def validate_endovis2(model, iters=32, mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    aug_params = {}
    status = 'val'
    root_dir= "/vol/bitbucket/cvb21/scared_accurate_3ds"
    save_path= "/vol/bitbucket/cvb21/pred_disp"
    file_dir = "/vol/"
    MIN_DEPTH = 5
    MAX_DEPTH = 250
    if os.path.exists(save_path)!=True:
        os.makedirs(save_path)
        
    keyframes = 5
    scared_d = [8, 9]
    out_list, epe_list = [], []
    errors = []
    ABE = []
    SCALE = []
    for i in scared_d:
        for k in range(keyframes):
            image1 = load_image("{}/dataset{}_keyframe{}/Left_Image_rect.png".format(root_dir, i,k))
            image2 = load_image("{}/dataset{}_keyframe{}/Right_Image_rect.png".format(root_dir, i, k))
            
            depth_gt_left = cv2.imread("{}/dataset{}_keyframe{}/left_depth_map_rect.bmp".format(root_dir, i, k), -1)

            with autocast(enabled=mixed_prec):
                disp = model(image1, image2, iters=iters, test_mode=True)
            disps_upsample = F.interpolate(disp[1][:, 0, :, :].unsqueeze(1), [1024, 1280], mode="bilinear", align_corners=False).squeeze().cpu().detach().numpy() # ask if this is really the right output value
            
            baseline, focal = get_bf(i)
            depth_pred = (baseline * focal) / (disps_upsample*1280)
            
            # save pred
            img_idx = f"{i}" + f"{k}"
            vmax_pred = np.percentile(depth_pred, 95)
            normalizer_pred = mpl.colors.Normalize(vmin=depth_pred.min(), vmax=vmax_pred)
            mapper_pred = mpl.cm.ScalarMappable(norm=normalizer_pred, cmap='magma')
            colormapped_im_pred = (mapper_pred.to_rgba(depth_pred)[:, :, :3]*255).astype(np.uint8)
            im_pred = Image.fromarray(colormapped_im_pred)
            dest_pred = os.path.join(save_path, '{}_pred.jpeg'.format(img_idx))
            im_pred.save(dest_pred)
            
            #save gt
            vmax_gt = np.percentile(depth_gt_left, 95)
            normalizer_gt = mpl.colors.Normalize(vmin=depth_gt_left.min(), vmax=vmax_gt)
            mapper_gt = mpl.cm.ScalarMappable(norm=normalizer_gt, cmap='magma')
            colormapped_im_gt = (mapper_gt.to_rgba(depth_gt_left)[:, :, :3] * 255).astype(np.uint8)
            im_gt = Image.fromarray(colormapped_im_gt)
            dest_gt = os.path.join(save_path, '{}_gt.jpeg'.format(img_idx))
            im_gt.save(dest_gt)

            # compute error
            mask = np.logical_and(depth_gt_left > MIN_DEPTH, depth_gt_left < MAX_DEPTH)
            depth_pred = depth_pred[mask]
            depth_gt_left = depth_gt_left[mask]
            depth_pred[depth_pred < MIN_DEPTH] = MIN_DEPTH
            depth_pred[depth_pred > MAX_DEPTH] = MAX_DEPTH
            difference_image = np.abs(depth_pred - depth_gt_left)
            abe = difference_image.mean()
            ABE.append(abe)
            errors.append(compute_errors(depth_gt_left, depth_pred, MIN_DEPTH, MAX_DEPTH))
            mean_errors = np.array(errors).mean(0)

    print('Mean Absolute Error:', np.mean(ABE))
    print('std of Absolute Error:', np.std(ABE))
    print('Finished Calculating Absolute Error')
    #### 7 criteria ####
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done Finished Testing!")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True, choices=["eth3d", "kitti", "things", "endovis"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=use_mixed_precision)

    elif args.dataset == 'things':
        validate_things(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
    elif args.dataset == "endovis":
        validate_endovis2(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
