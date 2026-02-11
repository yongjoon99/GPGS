# from tools import run_net

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'

from tools import test_net, test_net_infer, test_net_iter
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter

def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger = logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        config.dataset.val.others.bs = 1
        config.dataset.test.others.bs = 1
    else:
        config.dataset.train.others.bs = config.total_bs
        config.dataset.val.others.bs = 1
        config.dataset.test.others.bs = 1
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    args.seed=14356
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() 


    config.vis = args.vis

    config.dataset.test._base_.infer = args.infer
    config.model.transformer_config.mask_type = args.mask_type
    config.model.transformer_config.mask_type_iter = args.mask_type_iter
    config.model.transformer_config.mask_ratio = args.mask_ratio
    config.model.transformer_config.mask_ratio_iter = args.mask_ratio_iter
    config.dataset.test._base_.output_path = os.path.abspath(f'../output/{args.dataset}/{args.scene}')
    config.dataset.test._base_.center_coord = os.path.abspath(f'../output/{args.dataset}/{args.scene}/center_coord.npy')
    config.dataset.test._base_.DATA_PATH = os.path.abspath(f'../data/{args.dataset}/{args.scene}')
    config.dataset.test._base_.PC_PATH = os.path.abspath(f'../data/{args.scene}' )

    if config.dataset.test._base_.infer:

        test_net_infer(args,config)
    else:
        test_net(args, config)



if __name__ == '__main__':
    main()
