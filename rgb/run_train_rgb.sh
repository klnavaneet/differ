# Sample arguments to run training code. Set argument values as necessary.
# Check the list of arguments in train.py for more options

python train_rgb.py \
	--exp exp_dir \
	--gpu 0 \
	--category chair \
	--N_VIEWS 4 \
	--SIGMA_SQ_MASK 0.4 \
	--LOSS_MASK \
	--wt_bce 1.\
	--wt_aff_fwd 1. \
	--wt_aff_bwd 1. \
	--wt_rgb 1. \
	--skipconn \
	--print_n 100 \
	--save_n 1000 \
	--save_model_n 5000 \
	--N_ITERS 200000
