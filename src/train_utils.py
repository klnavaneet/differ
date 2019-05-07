# General Imports
import argparse
import glob
import json
import math
import os
import pprint
import random
import re
import sys
import time
import pdb

from itertools import product
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename

# General DL, CV imports
import cv2
import numpy as np
import scipy
import scipy.io as sio
import scipy.misc as sc
import imageio
import tensorflow as tf
import tflearn

from scipy import misc
from scipy.spatial.distance import cdist as np_cdist

BASE_DIR = dirname(abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('../src')
sys.path.append('../src/utils_chamfer')

import tf_nndistance
from net import recon_net_large as recon_net
from net import recon_net_large_partseg as recon_net_partseg
from net import recon_net_tiny_rgb_skipconn as recon_net_rgb_skipconn
from data_loader import get_feed_dict, get_shapenet_drc_models, get_shapenet_drc_models_partseg, fetch_batch_drc, fetch_batch_pcl_rgb
from shapenet_taxonomy import *
from proj_codes import *
		 

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, 
        help='Name of Experiment')
parser.add_argument('--gpu', type=str, required=True, 
        help='GPU to use')
parser.add_argument('--USE_3D_LOSS', action='store_true', 
        help='Supply this parameter to use 3D Chamfer loss')
parser.add_argument('--bottleneck', type=int, default=128, 
        help='latent space size')
parser.add_argument('--category', type=str, required=True, 
        help='Category')
parser.add_argument('--rotate', action='store_true', 
        help='Supply this parameter to rotate the point cloud')
parser.add_argument('--gt_size', type=int, default=1024, 
        help='GT size for training. Choose from [1024,2048].')
parser.add_argument('--batch_size', type=int, default=1, 
        help='Batch Size during training')
parser.add_argument('--lr', type=float, default=0.00005, 
        help='Learning Rate')
parser.add_argument('--print_n', type=int, default=100, 
        help='print training losses every print_n iteration')
parser.add_argument('--save_n', type=int, default=1000, 
        help='save summary and outputs every save_n iteration')
parser.add_argument('--save_model_n', type=int, default=1000, 
        help='save the network weights after every save_model_n iterations')
parser.add_argument('--partseg', action='store_true', 
        help='Supply this parameter if you want to predict part segmentations')
parser.add_argument('--grid_h', type=int, default=64, 
        help='projection grid height')
parser.add_argument('--grid_w', type=int, default=64, 
        help='projection grid width')
parser.add_argument('--max_epoch', type=int, default=500, 
        help='Maximum number of training epochs')
parser.add_argument('--N_VIEWS', type=int, default=12, 
        help='Number of projections for loss calculation')
parser.add_argument('--N_PTS', type=int, default=1024, 
        help='Number of points in predicted pcl')
parser.add_argument('--IMG_H', type=int, default=64, 
        help='Input image height')
parser.add_argument('--IMG_W', type=int, default=64, 
        help='Input image width')
parser.add_argument('--LAMBDA_REG', type=float, default=0., 
        help='Weight for regularization loss')
parser.add_argument('--save_pcl', action='store_true', 
        help='Save point clouds during training')
parser.add_argument('--views', type=str, default='random', 
        help='random/fixed angles for projection')
parser.add_argument('--CORR', action='store_true', 
        help='use projection from corresponding view of image')
parser.add_argument('--N_ITERS', type=int, default=400000, 
        help='Number of iters to train')

# Mask
parser.add_argument('--SIGMA_SQ_MASK', type=float, default=0.5, 
        help='variance of gaussian in mask projection')
parser.add_argument('--wt_bce', type=float, default=0., 
        help='Weight for bce loss')
parser.add_argument('--wt_3d', type=float, default=0., 
        help='Weight for 3D chamfer loss')
parser.add_argument('--wt_aff_fwd', type=float, default=0., 
        help='Weight for forward affinity loss')
parser.add_argument('--wt_aff_bwd', type=float, default=0., 
        help='Weight for backward affinity loss')

# Depth
parser.add_argument('--LOSS_DEPTH', action='store_true', 
        help='Supply this parameter if you want to use depth loss, else ignore.')
parser.add_argument('--SIGMA_SQ', type=float, default=0.5, 
        help='print_n')
parser.add_argument('--wt_depth', type=float, default=10., 
        help='Weight for depth loss')

# Dataset
parser.add_argument('--DATA_PFCN', action='store_true', 
        help='Supply this parameter if you want to use PFCN 500 dataset instead\
        of full ShapeNet, else ignore.')

# Partseg
parser.add_argument('--LOSS_PARTSEG', action='store_true', 
        help='Supply this parameter if you want to use partseg loss, else ignore.')
parser.add_argument('--WELL_RADIUS', type=float, default=1.0, 
        help='Radius of depth well')
parser.add_argument('--BETA', type=float, default=100., 
        help='Beta for scaling depth value')
parser.add_argument('--wt_partseg', type=float, default=10., 
        help='Weight for partseg loss')
parser.add_argument('--n_cls', type=int, default=4, 
        help='Number of classes for category')

# RGB
parser.add_argument('--LOSS_RGB', action='store_true', 
        help='Supply this parameter if you want to use rgb loss, else ignore.')
parser.add_argument('--skipconn', action='store_true', 
        help='skip connections in network arch to transmit image info for 
        color prediction')
parser.add_argument('--color_space', type=str, default='rgb', 
        help='Choose from [rgb, hsv, lab]')
parser.add_argument('--wt_rgb', type=float, default=10., 
        help='Weight for rgb loss')

# 2d metrics
parser.add_argument('--snapshot', type=str, 
        help='Load snapshot : ["<epoch>" ,"best_emd", "best_chamfer"]')
parser.add_argument('--model', type=str, 
        help='Method of rendering : ["differ", "capnet"]')
parser.add_argument('--feature', type=str, 
        help='to load partseg/rgb model')
parser.add_argument('--psgn', action='store_true', 
        help='to load psgn model instead of differ')

args = parser.parse_args()

print '-='*50
print args
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

if args.dataset == 'shapenet':
    data_dir = '../data/ShapeNet_rendered'
    data_dir_pcl = '../data/ShapeNet_pcl'
else args.dataset:
    data_dir = '../data/partseg/ShapeNet_rendered'
    data_dir_pcl = '../data/partseg/ShapeNet_pcl'

random.seed(1024)
np.random.seed(1024)
tf.set_random_seed(1024)

VAL_BATCH_SIZE = 1
NUM_VIEWS = 10
NUM_VIEWS_PROJ = args.N_VIEWS


def cdist(a, b):
    diff = tf.expand_dims(a, axis=2) - tf.expand_dims(b, axis=1)
    dist_mat = tf.reduce_sum(tf.square(diff), axis=-1)
    return dist_mat

def grid_dist(grid_h, grid_w):
    '''
    Compute distance between every point in grid to every other point
    '''
    x, y = np.meshgrid(range(grid_h), range(grid_w), indexing='ij')
    grid = np.asarray([[x.flatten()[i],y.flatten()[i]] for i in range(len(x.flatten()))])
    grid_dist = np_cdist(grid,grid) 
    grid_dist = np.reshape(grid_dist, [grid_h, grid_w, grid_h, grid_w])
    return grid_dist


def labels2img(labels_map, N_CLS, is_onehot=False, pcl=False):
    '''
    Convert part-segmentation representation from labels to a colour image
    args:
        labels_map: float, (BS,H,W) or (BS,H,W,N_CLS+1)
                label map, either labels or one-hot encoding (includes background class)
                one-hot representation can be a probabilistic value
        N_CLS: int, number of classes
        is_onehot: boolean, 
                   True if input is in one-hot representation
    returns:
        out: float, (BS,H,W,3)
             output colour coded map
    '''
    if is_onehot:
        labels_map = np.argmax(labels_map, axis=-1) # 0 is background label in labels.npy
    cc = [[0,187,255,255],[60,255,0,255],[255,68,0,255],[187,0,255,255]]

    BS, H, W = (labels_map.shape)[:3]
    out = np.zeros(shape=(BS,H,W,3))
    cls_indices = [];
    for k in range(N_CLS+1):
        cls_indices.append(labels_map==k+1)
        out[cls_indices[k]==True] = cc[k]
    return out


def get_loss_proj(pred, gt, loss='bce', w=1., min_dist_loss=None, dist_mat=None):
    min_dist = None; min_dist_inv = None;
    if loss == 'bce':
            print '\nBCE Logits Loss\n'
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt, logits=pred)

    elif loss == 'bce_sparse':
            print '\nBCE Logits Loss\n'
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=pred)

    elif loss == 'weighted_bce':
            print '\nWeighted BCE Logits Loss\n'
            loss = tf.nn.weighted_cross_entropy_with_logits(targets=gt, logits=pred, 
                    pos_weight=0.5)
    elif loss == 'l2_sq':
            print '\nL2 Squared Loss\n'
            loss = (pred-gt)**2

    elif loss == 'l1':
            print '\nL1 Loss\n'
            loss = abs(pred-gt)

    elif loss == 'bce_prob':
            print '\nBCE Loss\n'
            epsilon = 1e-8
            loss = -gt*tf.log(pred+epsilon)*w - (1-gt)*tf.log(tf.abs(1-pred+epsilon))

    elif loss == 'cosine_dist':
            print '\nCosine Distance\n'
            #loss = tf.losses.cosine_distance(gt, pred, dim=-1)
            pred_norm = tf.sqrt(tf.reduce_sum(pred**2, axis=-1)+1e-8)
            loss = 1. - (tf.reduce_sum(gt*pred, axis=-1)/(pred_norm+1e-8))

    elif loss == 'iou':
            print '\nIoU metric\n'
            pred_idx = tf.argmax(pred, axis=3)
            loss = tf.metrics.mean_iou(gt, pred_idx, args.n_cls+1) # tuple of (iou, conf_mat)

    if min_dist_loss != None:
            dist_mat += 1.
            gt_mask = gt #+ (1.-gt)*1e6*tf.ones_like(gt)
            gt_white = tf.expand_dims(tf.expand_dims(gt,3),3)
            gt_white = tf.tile(gt_white, [1,1,1,args.grid_h,args.grid_w])

            pred_white = tf.expand_dims(tf.expand_dims(pred,3),3)
            pred_white = tf.tile(pred_white, [1,1,1,args.grid_h,args.grid_w])

            gt_white_th = gt_white + (1.-gt_white)*1e6*tf.ones_like(gt_white)
            dist_masked = gt_white_th * dist_mat * pred_white

            pred_mask = (pred_white) + ((1.-pred_white))*1e6*tf.ones_like(pred_white)
            dist_masked_inv = pred_mask * dist_mat * gt_white 

            min_dist = tf.reduce_min(dist_masked, axis=[3,4])
            min_dist_inv = tf.reduce_min(dist_masked_inv, axis=[3,4])
    return loss, min_dist, min_dist_inv


def scale(gt_pc, pr_pc): #pr->[-1,1], gt->[-1,1]
    ''' 
    Scale GT and predicted PCL to a bounding cube with edges from [-0.5,0.5] in
    each axis. 
    args:
            gt_pc: float, (BS,N_PTS,3); GT point cloud
            pr_pc: float, (BS,N_PTS,3); predicted point cloud
    returns:
            gt_scaled: float, (BS,N_PTS,3); scaled GT point cloud
            pred_scaled: float, (BS,N_PTS,3); scaled predicted point cloud
    '''
    pred = tf.cast(pr_pc, dtype=tf.float32)
    gt   = tf.cast(gt_pc, dtype=tf.float32)

    pred_clean = tf.clip_by_value(pred,-0.4,0.4)

    min_gt = tf.convert_to_tensor([tf.reduce_min(gt[:,:,i], axis=1) for i in xrange(3)])
    max_gt = tf.convert_to_tensor([tf.reduce_max(gt[:,:,i], axis=1) for i in xrange(3)])

    min_pr = tf.convert_to_tensor([tf.reduce_min(pred_clean[:,:,i], axis=1) for i in xrange(3)])
    max_pr = tf.convert_to_tensor([tf.reduce_max(pred_clean[:,:,i], axis=1) for i in xrange(3)])

    length_gt = tf.abs(max_gt - min_gt)
    length_pr = tf.abs(max_pr - min_pr)

    diff_gt = tf.reduce_max(length_gt, axis=0, keep_dims=True) - length_gt
    diff_pr = tf.reduce_max(length_pr, axis=0, keep_dims=True) - length_pr

    new_min_gt = tf.convert_to_tensor([min_gt[i,:] - diff_gt[i,:]/2. for i in xrange(3)])
    new_max_gt = tf.convert_to_tensor([max_gt[i,:] + diff_gt[i,:]/2. for i in xrange(3)])
    new_min_pr = tf.convert_to_tensor([min_pr[i,:] - diff_pr[i,:]/2. for i in xrange(3)])
    new_max_pr = tf.convert_to_tensor([max_pr[i,:] + diff_pr[i,:]/2. for i in xrange(3)])

    size_pr = tf.reduce_max(length_pr, axis=0)
    size_gt = tf.reduce_max(length_gt, axis=0)

    scaling_factor_gt = 1. / size_gt # 2. is the length of the [-1,1] cube
    scaling_factor_pr = 1. / size_pr

    box_min = tf.ones_like(new_min_gt) * -0.5

    adjustment_factor_gt = box_min - scaling_factor_gt * new_min_gt
    adjustment_factor_pr = box_min - scaling_factor_pr * new_min_pr

    pred_scaled = tf.transpose((tf.transpose(pred) * scaling_factor_pr)) + tf.reshape(tf.transpose(adjustment_factor_pr), (-1,1,3))
    gt_scaled   = tf.transpose((tf.transpose(gt) * scaling_factor_gt)) + tf.reshape(tf.transpose(adjustment_factor_gt), (-1,1,3))

    return gt_scaled, pred_scaled


def get_chamfer_metrics(gt_pcl, pred_pcl):
    '''
    Obtain chamfer distance between GT and predicted point clouds
    args:
            gt_pcl: float, (BS,N_PTS,3); GT point cloud
            pred_pcl: float, (BS,N_PTS,3); predicted point cloud
    returns:
            dists_forward: float, (); forward chamfer distance
            dists_backward: float, (); backward chamfer distance
            chamfer_distance: float, (); chamfer distance
    '''
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt_pcl, pred_pcl)
    dists_forward = tf.reduce_mean(dists_forward, axis=1) # (BATCH_SIZE,NUM_POINTS) --> (BATCH_SIZE)
    dists_backward = tf.reduce_mean(dists_backward, axis=1)
    chamfer_distance = dists_backward + dists_forward
    return dists_forward, dists_backward, chamfer_distance


def create_folder(folders_list):
    ''' 
    Create empty directory if it doesn't already exist
    args:
            folders_list: list of str; list of directory paths
    '''
    for folder in folders_list:
        if not exists(folder):
            os.makedirs(folder)


def get_average_from_dict(ip_dict_list):
    avg_list = []
    for ip_dict in ip_dict_list:
        avg_list.append(np.asarray([item for item in ip_dict.values()]).mean())
    return avg_list


def average_stats(val_mean, val_batch, iters):
    ''' 
    Update cumulative loss values
    args:
            val_mean: list of float; cumulative mean value
            val_batch: list of float; current value
            iters: iteration number
    returns:
            val_upd: list of float; updated cumulative mean values
    '''
    val_upd = [((item*iters)+batch_item)/(iters+1) for (item, batch_item) in\
            zip(val_mean, val_batch)]
    return val_upd

