
def prepare_Categorized_Imagenet(loader_real, n_classes, save_dir="categorized_classes"):
        os.makedirs(save_dir, exist_ok=True)
        categorized_data = [{'data': [], 'targets': []} for _ in range(n_classes)]
        class_counts = {cls: 0 for cls in range(n_classes)}
        total_counts = {cls: 0 for cls in range(n_classes)}
        for _, target in tqdm(loader_real):
            for t in target:
                total_counts[t.item()] += 1
        for img, target in tqdm(loader_real, desc="Categorizing Data"):
            for i in range(len(target)):
                cls = target[i].item()  #
                categorized_data[cls]['data'].append(img[i])  
                categorized_data[cls]['targets'].append(target[i])  
                class_counts[cls] += 1
                if class_counts[cls] == total_counts[cls]:
                    class_data = {
                        'data': torch.stack(categorized_data[cls]['data']),
                        'targets': torch.tensor(categorized_data[cls]['targets'])
                    }
                    class_save_path = os.path.join(save_dir, f"class_{cls}.pt")
                    torch.save(class_data, class_save_path)
                    print(f"Class {cls} data saved to {class_save_path}")
                    categorized_data[cls]['data'] = []
                    categorized_data[cls]['targets'] = []
        print(f"All class data saved to {save_dir}")

def main_worker(args):
    train_set, _ = load_resized_data(args.dataset,args.data_dir,size=args.size,nclass=args.nclass,load_memory=args.load_memory)
    loader_real = ClassDataLoader(train_set,
                                batch_size=512,
                                num_workers=64,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)
    if args.rank==0:
        prepare_Categorized_Imagenet(loader_real, args.nclass, save_dir=args.imagenet_prepath)
    dist.barrier()

    
if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    import argparse
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import torch.distributed as dist
    from argsprocessor.args import ArgsProcessor
    from data.dataloader import ClassDataLoader
    from utils.utils import load_resized_data
    from utils.init_script import init_script

    parser = argparse.ArgumentParser(description='Configuration parser')
    parser.add_argument('--debug',dest='debug',action='store_true',help='When dataset is very large , you should get it')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--run_mode',type=str,choices=['Condense', 'Evaluation',"Pretrain"],default='Condense',help='Condense or Evaluation')
    parser.add_argument('-a','--aug_type',type=str,default='color_crop_cutout',help='augmentation strategy for condensation matching objective')
    parser.add_argument('--init',type=str,default='mix',choices=['random', 'noise', 'mix', 'load'],help='condensed data initialization type')
    parser.add_argument('--load_path',type=str,default=None,help="Path to load the synset")
    parser.add_argument('--gpu', type=str, default = "0",required=True, help='GPUs to use, e.g., "0,1,2,3"')  # 设置为 str 类型
    parser.add_argument('-i', '--ipc', type=int, default=1,required=True, help='number of condensed data per class')
    parser.add_argument('--imagenet_prepath', type=str, required=True, help='After the Preparation ,you should change the imagenet_prepath which in the Config YAML')
    parser.add_argument('--tf32', action='store_true',default=True,help='Enable TF32')
    args = parser.parse_args()

    args_processor = ArgsProcessor(args.config_path)

    args = args_processor.add_args_from_yaml(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    init_script(args)

    main_worker(args)


