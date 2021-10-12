from run_GraphFormers import *
from pathlib import Path
from parameters import parse_args
import torch.multiprocessing as mp
if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    args = parse_args()

    args.train_data_path = './data/dblp_graph_data/train.tsv'
    args.valid_data_path='./data/dblp_graph_data/valid.tsv'
    args.test_data_path='./data/dblp_graph_data/test.tsv'
    args.world_size=2 # GPU number
    args.mode='train'
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    cont=False
    if 'train' in args.mode:
        print('-----------train------------')
        if args.world_size > 1:
            mp.freeze_support()
            mgr = mp.Manager()
            end = mgr.Value('b', False)
            mp.spawn(train,
                     args = (args,end,cont),
                         nprocs=args.world_size,
                         join=True)
        else:
            end = None
            train(0,args,end,cont)


    if 'test' in args.mode:
        args.load_ckpt_name="ckpt/GraphFormers-test.pt"
        args.world_size=1
        print('-------------test--------------')
        test(0,args)

