###############################################################################
# Training code for reconstructing colored point clouds
# Use run_train_rgb.sh for running the code
###############################################################################

import os, sys
sys.path.append('../src')
from train_utils import *
from utils_train import *

# to hide scipy deprecation warnings while saving outputs
def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

category_id = shapenet_category_to_id[args.category]


def create_feed_dict(models, indices, models_pcl, b, args):
    batch = get_feed_dict(models, indices, models_pcl, b, args)
    batch_ip, batch_gt_rgb, batch_gt_mask, batch_pcl_gt, batch_x, batch_y, batch_names = batch
    feed_dict = {img_inp: batch_ip,  proj_gt_mask: batch_gt_mask, 
                proj_gt_rgb: batch_gt_rgb, pcl_gt: batch_pcl_gt,
                view_x:batch_x, view_y: batch_y}
    return feed_dict, batch_names

def get_epoch_loss(models, indices, models_pcl, args):
    batches = len(indices)/VAL_BATCH_SIZE
    val_loss, val_fwd, val_bwd = [0.]*3
    L_rgb_val, L_bce_val, L_fwd_val, L_bwd_val = [0.]*4
    
    for b in xrange(0,batches,10):
        feed_dict, _ = create_feed_dict(models, indices, models_pcl, b, args)
        L,F,B,_L_rgb_val, _L_bce_val,_L_fwd_val,_L_bwd_val = sess.run([chamfer_distance_scaled,
            dists_forward_scaled, dists_backward_scaled, loss_rgb, loss_bce,
            loss_fwd, loss_bwd], feed_dict)
        val_loss += L/batches
        val_fwd += F/batches
        val_bwd += B/batches
    batch_out = [_L_rgb_val, _L_bce_val, _L_fwd_val, _L_bwd_val]
    L_rgb_val, L_bce_val, L_fwd_val, L_bwd_val = get_average_from_dict(batch_out)
    return val_loss[0], val_fwd[0], val_bwd[0], L_rgb_val, L_bce_val, L_fwd_val, L_bwd_val


def save_outputs(models, pair_indices, models_pcl, global_step, args):
    sample_pair_indices = pair_indices[:10]
    out_dir_imgs = join(proj_images_folder, str(global_step))
    out_dir_pcl = join(proj_pcl_folder, str(global_step))
    create_folder([out_dir_imgs, out_dir_pcl])

    batches = len(sample_pair_indices)//VAL_BATCH_SIZE
    for b in xrange(batches):
        fd, model_names = create_feed_dict(models, pair_indices, models_pcl, b, args)

        # save projections
        _proj_mask, _proj_rgb = sess.run([proj_pred_mask, proj_pred_rgb], 
                feed_dict=fd)  
        for k in range(1): # view num 
            for l in range(1): # batch num
                _proj_mask[k][l][_proj_mask[k][l]>=0.5] = 1.
                _proj_mask[k][l][_proj_mask[k][l]<0.5] = 0.

                sc.imsave('%s/%s_%s_input.png'%(out_dir_imgs,model_names[l],k),
                        fd[img_inp][l])
                sc.imsave('%s/%s_%s_gt_mask.png'%(out_dir_imgs,model_names[l],k),
                        fd[proj_gt_mask][l][k])
                sc.imsave('%s/%s_%s_pred_mask.png'%(out_dir_imgs,model_names[l],k),
                        _proj_mask[k][l])

                batch_gt_rgb = (fd[proj_gt_rgb][k][l]*255.).astype(np.uint8)
                batch_gt_rgb = cv2.cvtColor(batch_gt_rgb, cv2.COLOR_RGB2BGR)
                batch_pred_rgb = (_proj_rgb[k][l]*255.).astype(np.uint8)
                batch_pred_rgb = cv2.cvtColor(batch_pred_rgb, cv2.COLOR_RGB2BGR)

                cv2.imwrite('%s/%s_%s_gt_rgb.png'%(out_dir_imgs,model_names[l],k), batch_gt_rgb)
                cv2.imwrite('%s/%s_%s_pred_rgb.png'%(out_dir_imgs,model_names[l],k), batch_pred_rgb)

        if args.save_pcl:
            # save pointclouds
            _pcl_out = sess.run(pcl_out, feed_dict=feed_dict)
            for k in range(1): #range(len(_pcl_out)):
                np.savetxt('%s/%s_%s_pred.xyz'%(out_dir_pcl,model_names[k],k),_pcl_out[k])
                np.savetxt('%s/%s_%s_gt.xyz'%(out_dir_pcl,model_names[k],k),batch_pcl_gt[k])


if __name__=='__main__':

    EXP_DIR = args.exp
    create_folder([EXP_DIR])
    filename = basename(__file__)
    os.system('cp %s %s'%(filename, EXP_DIR))

    # Define Logs Directories
    snapshot_folder = join(EXP_DIR, 'snapshots')
    best_folder = join(EXP_DIR, 'best')
    logs_folder = join(EXP_DIR, 'logs')
    log_file = join(EXP_DIR, 'logs.txt')
    proj_images_folder = join(EXP_DIR, 'log_proj_images')
    proj_pcl_folder = join(EXP_DIR, 'log_proj_pcl')
    
    # Create log directories
    create_folder([snapshot_folder, logs_folder, best_folder,
        proj_images_folder, proj_pcl_folder, join(snapshot_folder,'best')])
    
    args_file = join(logs_folder, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    train_models, val_models, train_pair_indices, val_pair_indices = get_shapenet_drc_models(data_dir, categs=[category_id])
    train_models_pcl, val_models_pcl, _, _ = get_shapenet_drc_models(data_dir_pcl, categs=[category_id])
    random.shuffle(val_pair_indices)

    batches = len(train_pair_indices) / args.batch_size
    
    # Create placeholders
    img_inp = tf.placeholder(tf.float32, shape=(args.batch_size, args.IMG_H, args.IMG_W, 3),
            name='img_inp')
    proj_gt_mask = tf.placeholder(tf.float32, shape=(args.batch_size, 
            args.N_VIEWS, args.IMG_H, args.IMG_W), name='proj_gt_mask')
    proj_gt_rgb = tf.placeholder(tf.float32, shape=(args.batch_size, 
            args.N_VIEWS, args.IMG_H, args.IMG_W, 3), name='proj_gt_rgb')

    pcl_gt = tf.placeholder(tf.float32, shape=(args.batch_size, args.N_PTS, 3), 
            name='pcl_gt_2K')
    view_x = tf.placeholder(tf.float32, shape=(args.batch_size,args.N_VIEWS), 
            name='view_x')
    view_y = tf.placeholder(tf.float32, shape=(args.batch_size,args.N_VIEWS), 
            name='view_y')

    # Tensorboard summary placeholders
    train_loss_summ = []
    loss_names = ['loss_total', 'loss_rgb', 'loss_bce', 'loss_aff_fwd', 'loss_aff_bwd']
    for idx, name in enumerate(loss_names):
        train_loss_summ.append(tf.placeholder(tf.float32, shape=(), name=name))

    val_loss_summ = []
    val_loss_names = ['chamfer_dist', 'val_loss_rgb', 'val_loss_bce', 
            'val_aff_fwd', 'val_aff_bwd', 'chamf_fwd', 'chamf_bwd']
    for idx, name in enumerate(val_loss_names):
        val_loss_summ.append(tf.placeholder(tf.float32, shape=(), name=name))

    # Build graph
    with tf.variable_scope('recon_net'):
        if args.skipconn:
            out = recon_net_rgb_skipconn(img_inp, args) # (B,N,3+C)
        else:
            out = recon_net_rgb(img_inp, args) # (B,N,3+C)
        pcl_out = out[:,:,:3] # (B,N,3)
        rgb_out = out[:,:,3:] # (B,N,3)
    
    '''
    pcl_out_rot --> dict of camera coordinate rotated pcls {(B,N,3)}
    pcl_out_persp --> dict of pcls after perspective_transform {(B,N,3)}
    proj_pred_mask --> dict of silhouette projection maps {(B,64,64)}
    proj_pred_rgb --> dict of rgb projection maps {(B,C,64,64)}
    proj_gt --> depth projection, placeholder of shape (B,V,64,64)
    proj_gt_mask --> mask projection, placeholder of shape (B,V,64,64)
    proj_gt_rgb --> placeholder of shape (B,V,64,64,3)
    
    loss --> final loss to optimize on
    loss_depth --> {V:(B,64,64)} --> l1 map for depth maps
    loss_bce --> {V:(B,64,64)} --> bce map for silhouette mask
    loss_rgb --> {V:(B,64,64)} --> bce map for rgb maps
    fwd --> {V:(B,64,64)} --> affinity loss fwd
    bwd --> {V:(B,64,64)} --> affinity loss bwd
    
    loss_fwd --> {()} --> reduce_mean values for each view
    loss_bwd --> {()} --> reduce_mean values for each view
    '''
    pcl_out_rot = {}; proj_pred_mask={}; proj_pred_rgb={}; pcl_out_persp = {}; loss = 0.;
    loss_bce = {}; fwd = {}; bwd = {}; loss_rgb = {}; prob = {}; prob_mask = {};
    loss_fwd = {}; loss_bwd = {};
    grid_dist_tensor = grid_dist(args.grid_h, args.grid_w)
    
    for idx in range(0,args.N_VIEWS):
        # 3D to 2D Projection
        pcl_out_rot[idx] = world2cam(pcl_out, view_x[:,idx], view_y[:,idx], args.batch_size) ### for batch size 1
        pcl_out_persp[idx] = perspective_transform(pcl_out_rot[idx], args.batch_size)
        proj_pred_mask[idx] = cont_proj(pcl_out_persp[idx], args.IMG_H, args.IMG_W, args.SIGMA_SQ_MASK)
        proj_pred_rgb[idx], prob[idx], prob_mask[idx] = rgb_cont_proj(pcl_out_persp[idx], rgb_out, args.N_PTS, args.IMG_H, args.IMG_W, args.WELL_RADIUS, args.BETA, 'rgb')

        # Loss
        # mask
        loss_bce[idx], fwd[idx], bwd[idx] = get_loss_proj(proj_pred_mask[idx],
                proj_gt_mask[:,idx], 'bce_prob', 1.0, True, grid_dist_tensor)
        loss_bce[idx] = tf.reduce_mean(loss_bce[idx])
        loss_fwd[idx] = 1e-4*tf.reduce_mean(fwd[idx])
        loss_bwd[idx] = 1e-4*tf.reduce_mean(bwd[idx]) 
        loss += args.wt_bce*loss_bce[idx] # add mask loss to main loss
        # add affinity loss if necessary
        loss += (args.wt_aff_fwd * loss_fwd[idx]) + \
                (args.wt_aff_bwd * loss_bwd[idx]) 

        # RGB reconstruction loss
        loss_rgb[idx], _, _ = get_loss_proj(proj_pred_rgb[idx],
                proj_gt_rgb[:,idx], 'l2_sq')
        loss_rgb[idx] = tf.reduce_mean(loss_rgb[idx])
        loss += args.wt_rgb*tf.reduce_mean(loss_rgb[idx])
    loss = (loss / args.N_VIEWS) 

    pcl_out_scaled, pcl_gt_scaled = scale(pcl_gt, pcl_out)
    dists_forward_scaled, dists_backward_scaled, chamfer_distance_scaled = get_chamfer_metrics(pcl_gt_scaled, pcl_out_scaled)
    
    train_vars = [var for var in tf.global_variables() if 'recon_net' in var.name]
    load_vars = [var for var in tf.global_variables() if 'Variable' not in var.name]
    
    # Optimizer
    opt = tf.train.AdamOptimizer(args.lr, beta1=0.9)
    optim = opt.minimize(loss, var_list=train_vars)

    # Training params
    start_epoch = 0
    max_epoch = args.max_epoch
    
    # Define savers to load and store models
    saver = tf.train.Saver(max_to_keep=2)
    saver_load = tf.train.Saver(load_vars)
    
    # Add Tensorboard summaries
    loss_summ = []
    for idx, name in enumerate(loss_names):
        loss_summ.append(tf.summary.scalar(name, train_loss_summ[idx]))
    train_summ = tf.summary.merge(loss_summ)

    # Add Tensorboard summaries
    loss_summ_val = []
    for idx, name in enumerate(val_loss_names):
        loss_summ_val.append(tf.summary.scalar(name, val_loss_summ[idx]))
    val_summ = tf.summary.merge(loss_summ_val)

    # GPU configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Run session
    with tf.Session(config=config) as sess:
        print 'Session started'
        train_writer = tf.summary.FileWriter(logs_folder+'/train', sess.graph_def)
        val_writer = tf.summary.FileWriter(logs_folder+'/val', sess.graph_def)

        print 'running initializer'
        sess.run(tf.global_variables_initializer())
        print 'done'

        # Load previous checkpoint
        init_flag = True
        ckpt = tf.train.get_checkpoint_state(snapshot_folder)
        if ckpt is not None:
            print ('loading '+os.path.abspath(ckpt.model_checkpoint_path) + '  ....')
            saver_load.restore(sess, os.path.abspath(ckpt.model_checkpoint_path))
            st_iters = int(re.match('.*-(\d*)$', ckpt.model_checkpoint_path).group(1))
            start_epoch = int(st_iters/batches)
            st_batches = st_iters % batches
            init_flag = False

        since = time.time()
        print '*'*30,'\n','Training Started !!!\n', '*'*30

        if start_epoch == 0:
            with open(log_file, 'w') as f:
                f.write(' '.join(['Epoch','Train_loss','Val_loss','Minutes','Seconds','\n']))

        train_loss_N, L_rgb_N, L_bce_N, L_fwd_N, L_bwd_N = [0.]*5
        batch_out_mean = [0.]*5
        epoch_out = [0.]*5
        best_val_loss = 1e5

        for i in xrange(start_epoch, max_epoch+1):
            random.shuffle(train_pair_indices)
            train_epoch_loss, train_epoch_rgb, train_epoch_bce, train_epoch_fwd, train_epoch_bwd = [0.]*5
            
            if init_flag:
                st_batches = 0

            for b in xrange(st_batches, batches):
                global_step = i*batches + b + 1
                if global_step >= args.N_ITERS:
                        sys.exit('Finished Training')
                
                # Load data
                feed_dict, _ = create_feed_dict(train_models, train_pair_indices,
                        train_models_pcl, b, args) 

                # Calculate loss and run optimizer 
                L, _L_rgb, _L_bce, _L_fwd, _L_bwd, _ = sess.run([loss, loss_rgb, 
                    loss_bce, loss_fwd, loss_bwd, optim], feed_dict)

                L_rgb, L_bce, L_fwd, L_bwd = get_average_from_dict([_L_rgb,
                    _L_bce, _L_fwd, _L_bwd])

                batch_out = [L, L_rgb, L_bce, L_fwd, L_bwd]
                # Use loss values averaged over N batches for logging
                batch_out_mean = average_stats(batch_out_mean, batch_out,
                        b%args.print_n)
                train_loss_N, L_rgb_N, L_bce_N, L_fwd_N, L_bwd_N = batch_out_mean
                epoch_out = average_stats(epoch_out, batch_out,
                        global_step%batches)

                if global_step % args.print_n == 0:
                    feed_dict_summ = {}
                    for idx, item in enumerate(batch_out_mean):
                        feed_dict_summ[train_loss_summ[idx]] = item

                    _summ = sess.run(train_summ, feed_dict_summ)
                    # Add to tensorboard summary
                    train_writer.add_summary(_summ, global_step)

                    time_elapsed = time.time() - since
                    _pcl_out = sess.run(pcl_out, feed_dict)

                    print 'Iter = {}  Loss = {:.5f}  RGB = {:.5f}  BCE = {:.5f}  FWD = {:.5f}  BWD = {:.5f}  Time = {:.0f}m {:.0f}s'.format(global_step, train_loss_N, L_rgb_N, L_bce_N, L_fwd_N, L_bwd_N, time_elapsed//60, time_elapsed%60)

                if (global_step-1) % args.save_n == 0:
                    save_outputs(val_models, val_pair_indices, val_models_pcl, 
                            global_step, args)

                if global_step % args.save_model_n == 0 and i != -1:
                    print 'Saving Model ....................'
                    saver.save(sess, join(snapshot_folder, 'model'), global_step=global_step)
                    print '..................... Model Saved'

                # Val metrics
                if global_step % args.save_n == 0:
                    val_t_st = time.time()
                    val_out = get_epoch_loss(val_models, val_pair_indices, 
                            val_models_pcl, args)
                    feed_dict_summ = {}
                    for idx, item in enumerate(val_out):
                        feed_dict_summ[val_loss_summ[idx]] = item
                    _summ = sess.run(val_summ, feed_dict_summ)

                    val_epoch_loss = val_out[0]
                    val_writer.add_summary(_summ, global_step)
                    val_t_sp = time.time()
                    print 'Val Epoch Loss: {:.4f}'.format(val_epoch_loss*10000)
                    
                    # Update best model if necessary
                    if (val_epoch_loss < best_val_loss):
                        saver.save(sess, join(snapshot_folder, 'best', 'best'))
                        os.system('cp %s %s'%(join(snapshot_folder, 'best/*'), best_folder))
                        best_val_loss = val_epoch_loss
                        print 'Best model at iter %s saved, loss = %.4f' %(global_step, best_val_loss*10000) 

            time_elapsed = time.time() - since
            with open(log_file, 'a') as f:
                    epoch_str = '{} {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f} {:.0f} {:.0f}'.format(i, 
                            train_epoch_loss, train_epoch_rgb, 
                            train_epoch_bce, train_epoch_fwd, train_epoch_bwd, 
                            time_elapsed//60, time_elapsed%60)
                    f.write(epoch_str+'\n')

            print '-'*65 + ' EPOCH ' + str(i) + ' ' + '-'*65
            epoch_str = 'TRAIN Loss: {:.6f}  RGB: {:.6f}  BCE: {:.6f}  FWD: {:.6f}  BWD: {:.6f}  Time:{:.0f}m {:.0f}s'.format(\
                    train_epoch_loss, train_epoch_rgb, 
                    train_epoch_bce, train_epoch_fwd, train_epoch_bwd, 
                    time_elapsed//60, time_elapsed%60)
            print epoch_str
            print '-'*140
            print

