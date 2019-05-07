import tensorflow as tf
import numpy as np
import pdb


def cont_proj(pcl, grid_h, grid_w, sigma_sq=0.5):
    '''
    Continuous approximation of Orthographic projection of point cloud
    to obtain Silhouette
    args:
            pcl: float, (N_batch,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
            grid_h, grid_w: int, ();
                     output depth map height and width
    returns:
            grid_val: float, (N_batch,H,W); 
                      output silhouette
    '''
    x, y, z = tf.split(pcl, 3, axis=2)
    pcl_norm = tf.concat([x, y, z], 2)
    pcl_xy = tf.concat([x,y], 2)
    out_grid = tf.meshgrid(tf.range(grid_h), tf.range(grid_w), indexing='ij')
    out_grid = [tf.to_float(out_grid[0]), tf.to_float(out_grid[1])]
    grid_z = tf.expand_dims(tf.zeros_like(out_grid[0]), axis=2) # (H,W,1)
    grid_xyz = tf.concat([tf.stack(out_grid, axis=2), grid_z], axis=2)  # (H,W,3)
    grid_xy = tf.stack(out_grid, axis=2)                # (H,W,2)
    grid_diff = tf.expand_dims(tf.expand_dims(pcl_xy, axis=2), axis=2) - grid_xy # (BS,N_PTS,H,W,2) 
    grid_val = apply_kernel(grid_diff, sigma_sq)    # (BS,N_PTS,H,W,2) 
    grid_val = grid_val[:,:,:,:,0]*grid_val[:,:,:,:,1]  # (BS,N_PTS,H,W) 
    grid_val = tf.reduce_sum(grid_val, axis=1)          # (BS,H,W)
    grid_val = tf.nn.tanh(grid_val)
    return grid_val


def apply_kernel(x, sigma_sq=0.5):
    '''
    Get the un-normalized gaussian kernel with point co-ordinates as mean and 
    variance sigma_sq
    args:
            x: float, (BS,N_PTS,H,W,2); mean subtracted grid input 
            sigma_sq: float, (); variance of gaussian kernel
    returns:
            out: float, (BS,N_PTS,H,W,2); gaussian kernel
    '''
    out = (tf.exp(-(x**2)/(2.*sigma_sq)))
    return out


def apply_ideal_kernel_depth(x, N_PTS, well_radius=0.5):
    out = tf.where(tf.abs(x)<=well_radius, tf.ones_like(x), 10*tf.ones_like(x))
    return out


def get_depth(pcl, grid_h, grid_w, N_PTS, well_radius=0.5):
    '''
    Well function for obtaining depth of every 3D input point at every 2D pixel
    args:
            pcl: float, (N_batch,N_Pts,3); input point cloud values assumed to
                                            be in (-1,1)
            grid_h, grid_w: int, (); output depth map height and width
            N_PTS: float, (); number of point in point cloud
            well_radius: float, (); radius of the depth well kernel
    return
            depth: float, (N_batch,N_Pts,H,W); output depth
    '''
    x, y, z = tf.split(pcl, 3, axis=2)
    pcl_norm = tf.concat([x, y, z], 2)
    pcl_xy = tf.concat([x,y], 2)
    out_grid = tf.meshgrid(tf.range(grid_h), tf.range(grid_w), indexing='ij')
    out_grid = [tf.to_float(out_grid[0]), tf.to_float(out_grid[1])]
    grid_z = tf.expand_dims(tf.zeros_like(out_grid[0]), axis=2) # (H,W,1)
    grid_xyz = tf.concat([tf.stack(out_grid, axis=2), grid_z], axis=2)  # (H,W,3)
    grid_xy = tf.stack(out_grid, axis=2)    # (H,W,2)
    grid_diff = tf.expand_dims(tf.expand_dims(pcl_xy, axis=2), axis=2) - grid_xy    # (BS,N_PTS,H,W,2) 
    grid_val = apply_ideal_kernel_depth(grid_diff, N_PTS, well_radius)    # (BS,N_PTS,H,W,2) 
    grid_val = grid_val[:,:,:,:,0]*grid_val[:,:,:,:,1]*tf.expand_dims(z,3)  # (BS,N_PTS,H,W) 
    depth = tf.clip_by_value(grid_val,0.,10.)
    return depth


def get_proj_prob_exp(d, beta=5., N_PTS=1024, ideal=False):
    '''
    Probability of a point being projected at each pixel of the projection
    map
    args:
        d: depth value of each point when projected at each pixel of projection
           (N_batch,N_PTS,grid_h,grid_w). This value is between 0 and 10. For
           points that are within 0.5 distance from grid point, it is min(0,z),
           for the rest, it is max(10,z).
    returns:
        prob: probablility of point being projected at each pixel
              float, (N_batch,N_PTS,grid_h,grid_w)    
    '''
    d_inv = 1. / (d+1e-5)
    if ideal:
        # ideal projection probabilities - prob=1 for min depth, 0 for rest
        prob = tf.transpose(tf.one_hot(tf.argmax(d_inv, axis=1), N_PTS), [0,3,1,2])
    else:
        prob = tf.nn.softmax(d_inv*beta, dim=1) # for every pixel, apply softmax across all points
    return prob


def rgb_cont_proj(pcl, feat, N_PTS, grid_h, grid_w, well_radius, beta, mode):
    '''
    2D Projection of any general feature of 3D point cloud
    args:
            pcl: float, (N_batch,N_Pts,3); input point cloud
                     values assumed to be in (-1,1)
            feat: float, (N_batch, N_Pts, N_cls) 
            N_PTS: int, (); Number of points in PCL
            grid_h, grid_w: int, (); output depth map height and width
            well_radius: radius of depth well beyond which to mask out 
                         probabilities
            mode: str, Choose between ['rgb','partseg']
    returns:
            proj_feat: float, (N_batch,H,W,N_cls+1) output feature map 
                                including background label at position 0
            prob: probablility of point being projected at each pixel
                  (N_batch,N_PTS,grid_h,grid_w)   
            mask: bool, (BS,H,W); mask of projection
    '''
    add_depth_range = tf.constant([0,0,1], dtype=tf.float32) # z dim changed from [-1,1] to [0,2]. needed for getting the correct probabilities.
    depth_val = get_depth(pcl+add_depth_range, grid_h, grid_w, N_PTS, well_radius) # (BS,N_PTS,H,W)
    prob = get_proj_prob_exp(depth_val, beta) # (BS,N_PTS,H,W)
    # Mask out the regions where no point is projected
    mask = tf.logical_not(tf.equal(10.*tf.ones_like(depth_val, tf.float32), depth_val)) # (BS,N_Pts,H,W)
    mask = tf.cast(mask, tf.float32)
    prob = prob*mask
    # Normalize probabilities
    prob = prob/(tf.reduce_sum(prob, axis=1, keep_dims=True) + 1e-8)
    # Expectation of feature values
    proj_feat = tf.expand_dims(prob, axis=-1) * (tf.expand_dims(tf.expand_dims(tf.to_float(feat),
        axis=2), axis=2)) # (BS,N_PTS,H,W,N_cls)
    proj_feat = tf.reduce_sum(proj_feat, axis=1) # (BS,H,W,N_cls) --> one-hot
    # mask out background i.e. regions where all point contributions sum to 0
    BS,H,W,_ = [int(d) for d in proj_feat.shape]
    mask = tf.reduce_sum(mask, axis=1) # (BS,H,W)
    if mode == 'partseg':
        # Insert background label at position 0
        mask = tf.cast(tf.equal(tf.zeros_like(mask), mask), tf.float32) # (BS,H,W)
        bgnd_lbl = tf.ones(shape=(BS,H,W,1)) * tf.expand_dims(mask,axis=-1) #(BS,H,W,1)
        proj_feat = tf.concat([bgnd_lbl,proj_feat], axis=-1) #(BS,H,W,N_cls+1)
    elif mode == 'rgb' or mode =='normals':
        # remove color/normals from background regions
        mask = tf.cast(tf.logical_not(tf.equal(tf.zeros_like(mask), mask)), tf.float32) # (BS,H,W)
        proj_feat = proj_feat * tf.expand_dims(mask,axis=-1)
    return proj_feat, prob, mask


def perspective_transform(xyz, batch_size):
    '''
    Perspective transform of pcl; Intrinsic camera parameters are assumed to be
    known (here, obtained using parameters of GT image renderer, i.e. Blender)
    Here, output grid size is assumed to be (64,64) in the K matrix
    TODO: use output grid size as argument
    args:
            xyz: float, (BS,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
    returns:
            xyz_out: float, (BS,N_PTS,3); perspective transformed point cloud 
    '''
    K = np.array([
            [120., 0., -32.],
            [0., 120., -32.],
            [0., 0., 1.]]).astype(np.float32)
    K = np.expand_dims(K, 0)
    K = np.tile(K, [batch_size,1,1])

    xyz_out = tf.matmul(K, tf.transpose(xyz, [0,2,1]))
    xy_out = xyz_out[:,:2]/abs(tf.expand_dims(xyz[:,:,2],1))
    xyz_out = tf.concat([xy_out, abs(xyz_out[:,2:])],axis=1)
    return tf.transpose(xyz_out, [0,2,1])


def world2cam(xyz, az, el, batch_size, N_PTS=1024):
    '''
    Convert pcl from world co-ordinates to camera co-ordinates
    args:
            xyz: float, (BS,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
            az: float, (BS); azimuthal angle of camera in radians
            elevation: float, (BS); elevation of camera in radians
            batch_size: int, (); batch size
            N_PTS: float, (); number of points in point cloud
    returns:
            xyz_out: float, (BS,N_PTS,3); output point cloud in camera
                        co-ordinates
    '''
    # Distance of object from camera - fixed to 2
    d = 2.
    # Calculate translation params
    # Camera origin calculation - az,el,d to 3D co-ord
    tx, ty, tz = [0, 0, d]
    rotmat_az=[
                [tf.ones_like(az),tf.zeros_like(az),tf.zeros_like(az)],
                [tf.zeros_like(az),tf.cos(az),-tf.sin(az)],
                [tf.zeros_like(az),tf.sin(az),tf.cos(az)]
                ]

    rotmat_el=[
                [tf.cos(el),tf.zeros_like(az), tf.sin(el)],
                [tf.zeros_like(az),tf.ones_like(az),tf.zeros_like(az)],
                [-tf.sin(el),tf.zeros_like(az), tf.cos(el)]
                ]

    rotmat_az = tf.transpose(tf.stack(rotmat_az, 0), [2,0,1])
    rotmat_el = tf.transpose(tf.stack(rotmat_el, 0), [2,0,1])
    rotmat = tf.matmul(rotmat_el, rotmat_az)

    tr_mat = tf.tile(tf.expand_dims([tx, ty, tz],0), [batch_size,1]) # [B,3]
    tr_mat = tf.expand_dims(tr_mat,2) # [B,3,1]
    tr_mat = tf.transpose(tr_mat, [0,2,1]) # [B,1,3]
    tr_mat = tf.tile(tr_mat,[1,N_PTS,1]) # [B,1024,3]

    xyz_out = tf.matmul(rotmat,tf.transpose((xyz),[0,2,1])) - tf.transpose(tr_mat,[0,2,1])

    return tf.transpose(xyz_out,[0,2,1])
