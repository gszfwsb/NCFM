def main_work(args):

    _,val_loader = get_loader(args)


    synset = Condenser(args, nclass_list=list(range(0,args.nclass)), nchannel=args.nch, hs=args.size, ws=args.size, device='cuda')

    for rank in range (args.world_size):
        if rank==args.rank:
            synset.load_condensed_data(loader=None, init_type="load",load_path=args.load_path)
        dist.barrier()

    syndataloader = synset.get_syndataLoader(args, args.augment)

    synset.continue_learning(args, syndataloader, val_loader)
    dist.destroy_process_group()


    
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from  utils.utils import get_loader
    from utils.init_script import init_script
    import argparse
    from argsprocessor.args import ArgsProcessor
    from condenser.Condenser import Condenser
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description='Configuration parser')
    parser.add_argument('--debug',dest='debug',action='store_true',help='When dataset is very large , you should get it')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--run_mode',type=str,choices=['Condense', 'Evaluation',"Pretrain"],default='Evaluation',help='Condense or Evaluation')
    parser.add_argument('--init',type=str,default='load',choices=['random', 'noise', 'mix', 'load'],help='condensed data initialization type')
    parser.add_argument('--load_path',type=str,required=True,help="Path to load the synset")
    parser.add_argument('--val_repeat',type=int,default=10,help='The times of validation on syn_dataset Imagenet only 3 times')
    parser.add_argument('--gpu', type=str, default = "0",required=True, help='GPUs to use, e.g., "0,1,2,3"')  # 设置为 str 类型
    parser.add_argument('-i', '--ipc', type=int, default=1, help='number of condensed data per class')
    parser.add_argument('--tf32', action='store_true',default=True,help='Enable TF32')
    parser.add_argument('--softlabel',dest='softlabel',action='store_true',help='Use the softlabel to evaluate the dataset')
    parser.add_argument('--step', type=int, default=5,required=True, help='number of condensed data per class')
    args = parser.parse_args()
    args_processor = ArgsProcessor(args.config_path)

    args = args_processor.add_args_from_yaml(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


    init_script(args)


    main_work(args)


