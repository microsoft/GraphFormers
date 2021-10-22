import os
from pathlib import Path

import torch.multiprocessing as mp

from src.parameters import parse_args
from src.run import train, test
from src.utils import setuplogging

if __name__ == "__main__":

    setuplogging()
    gpus = ','.join([str(_ + 1) for _ in range(2)])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    args = parse_args()
    print(os.getcwd())
    args.log_steps = 5
    args.world_size = 2  # GPU number
    args.mode = 'train'
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    cont = False
    if args.mode == 'train':
        print('-----------train------------')
        if args.world_size > 1:
            mp.freeze_support()
            mgr = mp.Manager()
            end = mgr.Value('b', False)
            mp.spawn(train,
                     args=(args, end, cont),
                     nprocs=args.world_size,
                     join=True)
        else:
            end = None
            train(0, args, end, cont)

    if args.mode == 'test':
        args.load_ckpt_name = "/data/workspace/Share/junhan/TopoGram_ckpt/dblp/topogram-pretrain-finetune-dblp-best3.pt"
        print('-------------test--------------')
        test(args)
