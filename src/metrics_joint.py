'''
Code for computing chamfer and emd metrics for the baseline and projection trained models.

Usage:
metrics:
visualization:
python metrics_joint.py --dataset saf --exp ./expts/saf/joint/multicat/3_full_data_joint_exp9_lrec_1e4/ --ip 124 --gpu 1 --category chair --eval_set test --snapshot best_joint --n_cls 4 --visualize
'''

from metrics_utils import *
#from train_utils import *
from net import recon_net_tiny_rgb_skipconn as joint_rgb_net
import pdb
import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, 
        help='Name of Experiment Prefixed with index')
parser.add_argument('--gpu', type=str, required=True, 
        help='GPU to use')
parser.add_argument('--dataset', type=str, required=True, 
        help='Choose from [shapenet, pfcn]')
parser.add_argument('--category', type=str, required=True, 
        help='Category to visualize from : ["airplane", "car", "chair"]')
parser.add_argument('--eval_set', type=str, required=True, 
        help='set to compute metrics on : ["train", "test"]')
parser.add_argument('--snapshot', type=str, required=True, 
        help='Load snapshot : ["<epoch>" ,"best_emd", "best_chamfer"]')
parser.add_argument('--batch_size', type=int, default=10, 
        help='Batch Size during evaluation. Make sure to set a value that\
        perfectly divides the total number of samples.')
parser.add_argument('--N_PTS', type=int, default=1024, 
        help='number of points in predicted point cloud')
parser.add_argument('--bottleneck', type=int, default=128, 
        help='dimension of encoder output')
parser.add_argument('--debug', action='store_true', 
        help='debug mode. only run the first 10 samples')
parser.add_argument('--freq_wt', action='store_true', 
        help='Calculate frequency weighted IOU')
parser.add_argument('--skipconn', action='store_true', 
        help='Provide if two branched used in network arch')
# visualize
parser.add_argument('--visualize', action='store_true', 
        help='visualize generated point clouds')
parser.add_argument('--save_screenshots', action='store_true', 
        help='save screenshots')
parser.add_argument('--save_gifs', action='store_true', 
        help='save gifs')
# misc
parser.add_argument('--tqdm', action='store_true', 
        help='view progress bar')
parser.add_argument('--mode', type=str, default='baseline', 
        help='For loading either 3D baseline model or projection based model.\
        Choose from [baseline, projection, only_mask]')
parser.add_argument('--color_space', type=str, default='rgb', 
        help='Use one of [rgb, hsv, lab] based on prediction color space')

args = parser.parse_args()

print '-='*50
print args
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

random.seed(1024)

data_dir = get_data_dir(args.dataset)
BATCH_SIZE = args.batch_size
HEIGHT = 64
WIDTH = 64
NUM_VIEWS = 10
args.N_PTS = 1024
ballradius = 3


if __name__=='__main__':

    exp_dir = os.path.abspath(args.exp)
    # Snapshot Folder Location
    snapshot_folder = join(BASE_DIR, exp_dir, 'snapshots')

    # use case
    if args.visualize:
        import show3d_balls
    elif args.save_screenshots or args.save_gifs:
        import show3d_balls
        screenshot_dir = join(exp_dir, 'screenshots', args.snapshot, args.category)
        create_folder([screenshot_dir])
    else:
        if exp_dir == '':
            print 'exp name is empty! Check code.'
        csv_path = join(exp_dir, '%s_%s.csv'%(args.eval_set, args.snapshot))
        with open(csv_path, 'w') as f:
            f.write('Id; Seg; Chamfer; Fwd; Bwd; Emd\n')

    # placeholders
    img_inp = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, 3), 
            name='img_inp')
    pcl_gt = tf.placeholder(tf.float32, shape=(None, args.N_PTS, 3), 
            name='pcl_gt')
    pred_pcl = tf.placeholder(tf.float32, shape=(None, args.N_PTS, 3), 
            name='pcl_pred')
    rgb_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, args.N_PTS, 3),
            name='rgb_gt')
    rgb_pred = tf.placeholder(tf.float32, shape=(BATCH_SIZE, args.N_PTS, 3),
            name='rgb_pred')

    # Build graph
    if args.mode=='baseline':
        with tf.variable_scope('joint_rgb_net'):
            pred_pcl, rgb_pred = joint_rgb_net(img_inp, args)
    elif args.mode=='projection':
        with tf.variable_scope('recon_net'):
            pred = joint_rgb_net(img_inp, args)
            pred_pcl, rgb_pred = tf.split(pred, 2, axis=-1)

    # metrics - reconstruction 
    gt_pcl_scaled, pred_pcl_scaled = scale(pcl_gt, pred_pcl)
    dists_forward, dists_backward, chamfer_distance, emd = get_rec_metrics(gt_pcl_scaled, pred_pcl_scaled)
    
    # metrics - rgb
    # Chamfer 
    pts_match_fwd, pts_match_bwd = get_labels_seg(pred_pcl_scaled, gt_pcl_scaled,
            'chamfer')
    rgb_gt_match = tf.stack([tf.gather(rgb_gt[k], pts_match_fwd[k], axis=0) \
                                for k in range(BATCH_SIZE)], axis=0)
    rgb_pred_match = tf.stack([tf.gather(rgb_pred[k], pts_match_bwd[k], axis=0) \
                                for k in range(BATCH_SIZE)], axis=0)

    rgb_loss_fwd, per_inst_fwd, per_inst_fwd_lbl = get_rgb_loss(rgb_pred, rgb_gt_match) 
    rgb_loss_bwd, per_inst_bwd, per_inst_bwd_lbl = get_rgb_loss(rgb_pred_match, rgb_gt)
    per_instance_rgb_loss = per_inst_fwd + per_inst_bwd
    rgb_loss = rgb_loss_fwd + rgb_loss_bwd

    # GPU configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Run session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # saver to load prevrgbs checkpoint
        saver = tf.train.Saver()
        load_previous_checkpoint(snapshot_folder, saver, sess,
                exp_dir, args.snapshot)
        tflearn.is_training(False, session=sess)

        # data
        if args.eval_set == 'train':
                models,indices = get_drc_models(data_dir, args.category, NUM_VIEWS, 'train')
        elif args.eval_set == 'val':
                models,indices = get_drc_models(data_dir, args.category, NUM_VIEWS, 'val')
        elif args.eval_set == 'test':
                models,indices = get_drc_models(data_dir, args.category, NUM_VIEWS, 'test')
        else:
                print 'Enter train/val/test for eval_set'
                sys.exit(1)

        if args.visualize or args.save_screenshots or args.save_gifs:
                random.shuffle(indices)
        batches = len(indices) // BATCH_SIZE

        print('computing metrics for %d samples...'%len(indices))
        if args.debug:
                iters = 50
        else:
                iters = range(batches)
        if args.tqdm:
                iters = tqdm(iters)

        rgb_parts_all = []
        n_err = 0
        for cnt in iters:
            try:
                # load batch
                batch_ip, batch_gt, batch_lbl = fetch_batch_joint(models, indices, cnt, BATCH_SIZE)
                fids = fetch_batch_paths(models, indices, cnt, BATCH_SIZE)
                feed_dict={img_inp:batch_ip, pcl_gt:batch_gt, 
                        rgb_gt: batch_lbl}
                _pred_pcl, _rgb_pred = sess.run([pred_pcl, rgb_pred], feed_dict)

                _pi_rgb_loss, _pi_seg_pred = sess.run([per_instance_rgb_loss,
                    per_inst_fwd_lbl], feed_dict)
                _pi_seg_pred[_pi_seg_pred<0] = 0.

                # metrics
                # C,F,B,E are all arrays of dimension (BATCH_SIZE,)
                if args.visualize or args.save_screenshots:
                    _gt_scaled, _pred_scaled = sess.run([gt_pcl_scaled, 
                        pred_pcl_scaled], feed_dict)
                else:
                    _pred_pcl = sess.run(pred_pcl, feed_dict)
                    _pred_pcl = tf_rotate(tf_rotate(_pred_pcl,0,90).eval(),90,0).eval()
                    _gt_scaled, _pred_scaled = sess.run([gt_pcl_scaled, 
                        pred_pcl_scaled], feed_dict={pcl_gt:batch_gt, 
                            pred_pcl:_pred_pcl})

                C,F,B,E = sess.run([chamfer_distance, dists_forward, 
                    dists_backward, emd], feed_dict={gt_pcl_scaled:_gt_scaled, 
                        pred_pcl_scaled:_pred_scaled})
                S = _pi_rgb_loss

                # visualize
                if args.visualize:
                    gt = tf_rotate(_gt_scaled,-90,-90).eval()
                    if args.mode=='baseline':
                        pr = tf_rotate(_pred_scaled,-90,-90).eval()
                    else:
                        pr = _pred_scaled
                    for b in xrange(BATCH_SIZE):
                        # Image RGB2BGR
                        batch_ip[b] = np.flip(batch_ip[b], axis=-1)
                        print 'Model:{} C:{:.6f} F:{:.6f} B:{:.6f} E:{:.6f}'.format(fids[b],C[b],F[b],B[b],E[b])
                        cv2.imshow('', batch_ip[b])
                        saveBool = show3d_balls.showtwopoints(gt[b], pr[b], ballradius=ballradius)
                        
                        _pcl_gt = np_rotate(batch_gt[b],-90*np.pi/180.,-90*np.pi/180.)
                        if args.mode=='baseline':
                            pcl_pred = np_rotate(_pred_pcl[b],-90*np.pi/180.,-90*np.pi/180.)
                        else:
                            pcl_pred = _pred_pcl[b]
    
                        # Avoid some small negative values, if present, from   
                        # being mapped to 255 by uint8 conversion. 
                        _pi_seg_pred[b][_pi_seg_pred[b]<0] = 0.

                        print 'Model:{} rgb_loss:{}'.format(fids[b], _pi_rgb_loss[b])
                        show3d_balls.showpoints_partseg(_pcl_gt, 
                                batch_lbl[b]*255., ballradius=5)
                        show3d_balls.showpoints_partseg(pcl_pred,
                                _pi_seg_pred[b]*255., ballradius=5)

                # screenshots and gifs
                elif args.save_screenshots or args.save_gifs:
                    gt = tf_rotate(_gt_scaled,-90,-90).eval()
                    if args.mode=='baseline':
                        pr = tf_rotate(_pred_scaled,-90,-90).eval()
                    else:
                        pr = _pred_scaled
                    # RGB2BGR
                    for b in xrange(BATCH_SIZE):
                        batch_ip[b] = np.flip(batch_ip[b], axis=-1)
                        save_screenshots(gt[b], pr[b], batch_ip[b], 
                            screenshot_dir, fids[b], args.eval_set, 7, 
                            args, True, batch_lbl[b]*255, _pi_seg_pred[b]*255)
                    print 'done'
                    if cnt == 3:
                        sys.exit()

                # save metrics to csv
                else:
                    if np.isnan(C).any() or np.isnan(E).any():
                        print fids
                        print C
                        print E
                    else:
                        with open(csv_path, 'a') as f:
                            for b in xrange(BATCH_SIZE):
                                f.write('{};{:.6f};{:.6f};{:.6f};{:.6f};{:.6f}\n'.format(fids[b],S[b],C[b],F[b],B[b],E[b]))
            except KeyboardInterrupt:
                raise
            except:
                print fids
                n_err += 1
                continue

        # get avg metrics
        S_avg,C_avg,F_avg,B_avg,E_avg = get_averages(csv_path)
        print 'Final Metrics:  Chamfer  Forward  Backward EMD  Seg'.format(C_avg,F_avg,B_avg,E_avg,S_avg)
        print '{:.6f};  {:.6f};  {:.6f};  {:.6f};  {:.6f}'.format(C_avg,F_avg,B_avg,E_avg,S_avg)
        print 'Models with error: ', n_err
