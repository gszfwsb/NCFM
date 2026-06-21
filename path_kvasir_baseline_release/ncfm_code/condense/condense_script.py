def main_worker(args):
    
    args.class_list = distribute_class(args.nclass,args.debug)

    plotter = get_plotter(args)

    loader_real,_ = get_loader(args)


    aug, _ = diffaug(args)
    
    condenser = Condenser(args, nclass_list=args.class_list, nchannel=args.nch, hs=args.size, ws=args.size, device='cuda')
    for local_rank in range(args.local_world_size):
        if  args.local_rank == local_rank:
            condenser.load_condensed_data(loader_real, init_type=args.init,load_path=args.load_path)
            print(f"============RANK:{dist.get_rank()}====LOCAL_RANK {local_rank} Loaded Condensed Data==========================")
        dist.barrier()

    model_init,model_interval,model_final = get_feature_extractor(args)
    if getattr(args, "use_local_patch_feature_ncfd", False):
        if local_patch_uses_model_interval(args):
            args.local_patch_encoder = None
            args.local_patch_encoder_indices = []
            args.local_patch_encoder_stage = "model_interval"
            args.local_patch_encoder_mode = "model_interval_step"
            args.local_patch_encoder_paths = []
            args._local_patch_last_encoder_index = "model_interval"
        else:
            args.local_patch_encoder = build_frozen_patch_encoder(args)
        if args.rank == 0:
            args.logger(
                "Local patch-feature NCFD enabled: "
                f"grid={args.local_patch_grid}, "
                f"lambda={args.lambda_local_patch_ncfd}, "
                f"local_num_freqs={getattr(args, 'local_patch_num_freqs', 'auto')}, "
                f"source={getattr(args, 'local_patch_encoder_source', 'premodel0_trained')}, "
                f"indices={getattr(args, 'local_patch_encoder_indices', [])}, "
                f"blocks={getattr(args, 'local_patch_encoder_blocks', 2)}, "
                f"feature_dim_check={getattr(args, 'local_patch_feature_dim', 'auto')}"
            )
    else:
        args.local_patch_encoder = None
    optim_img = get_optimizer(optimizer=args.optimizer, parameters=condenser.parameters(),lr=args.lr_img, mom_img=args.mom_img,weight_decay=args.weight_decay,logger=args.logger)
    if args.sampling_net:
        feature_dim = infer_feature_dim(args, model_interval)
        sampling_net = SampleNet(feature_dim=feature_dim, t_batchsize=args.num_freqs).to(args.device)
        optim_sampling_net = get_optimizer(optimizer= "sgd", parameters=sampling_net.parameters(),lr=args.lr_sampling_net, mom_img=args.mom_img,weight_decay=args.weight_decay,logger=args.logger)
    else:
        sampling_net = None
        optim_sampling_net = None
    condenser.condense(args,plotter,loader_real,aug,optim_img,model_init,model_interval,model_final,sampling_net,optim_sampling_net)

    dist.destroy_process_group()


def infer_feature_dim(args, model):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, args.nch, args.size, args.size, device=args.device)
        _, feat = model(dummy, return_features=True)
    if was_training:
        model.train()
    if args.rank == 0:
        args.logger(f"SampleNet feature_dim={feat.shape[1]}, t_batchsize={args.num_freqs}")
    return feat.shape[1]



if __name__ == '__main__':
    import sys
    import os
    import torch
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.diffaug import diffaug
    import torch.distributed as dist
    from  utils.ddp import distribute_class
    from  utils.utils import get_plotter,get_optimizer,get_loader,get_feature_extractor
    from utils.init_script import init_script
    import argparse
    from argsprocessor.args import ArgsProcessor
    from condenser.Condenser import Condenser
    from condenser.local_patch_ncfd import (
        build_frozen_patch_encoder,
        local_patch_uses_model_interval,
    )
    from NCFM.SampleNet import SampleNet

    parser = argparse.ArgumentParser(description='Configuration parser')
    parser.add_argument('--debug',dest='debug',action='store_true',help='When dataset is very large , you should get it')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--run_mode',type=str,choices=['Condense', 'Evaluation',"Pretrain"],default='Condense',help='Condense or Evaluation')
    parser.add_argument('-a','--aug_type',type=str,default='color_crop_cutout',help='augmentation strategy for condensation matching objective')
    parser.add_argument('--init',type=str,default='mix',choices=['random', 'noise', 'mix', 'load'],help='condensed data initialization type')
    parser.add_argument('--load_path',type=str,default=None,help="Path to load the synset")
    parser.add_argument('--gpu', type=str, default = "0",required=True, help='GPUs to use, e.g., "0,1,2,3"') 
    parser.add_argument('-i', '--ipc', type=int, default=1,required=True, help='number of condensed data per class')
    parser.add_argument('--tf32', action='store_true',default=True,help='Enable TF32')
    args = parser.parse_args()
    args_processor = ArgsProcessor(args.config_path)

    args = args_processor.add_args_from_yaml(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    init_script(args)

    main_worker(args)
