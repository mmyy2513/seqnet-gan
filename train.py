import argparse
import datetime
import os.path as osp
import time

import torch
import torch.utils.data

from datasets import build_test_loader, build_train_loader
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch
from models.seqnet import SeqNet
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed

import sys
dgnet_path = "/root/workplace/PersonSearch/DG-Net"
sys.path.append(dgnet_path)

from util import get_config
from trainer import DGNet_Trainer, to_gray

import argparse
from torch.autograd import Variable

import torch
import os
import numpy as np
from torchvision import datasets
from PIL import Image
import cv2

from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

device = torch.device("cuda:1")

name = 'E0.5new_reid0.5_w30000'

config = get_config("config.yaml")

def main(args):
    cfg = get_default_cfg()
    
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model")
    model = SeqNet(cfg)
    model.to(device)
    
    if args.eval == False:
        ## GAN
        model_g = DGNet_Trainer(config)
        model_g.to(device)
        
        state_dict_gen = torch.load(dgnet_path+"/outputs/%s/checkpoints/gen_00100000.pt"%name)
        model_g.gen_a.load_state_dict(state_dict_gen['a'], strict=False)
        model_g.gen_b = model_g.gen_a

        state_dict_id = torch.load(dgnet_path+"/outputs/%s/checkpoints/id_00100000.pt"%name)
        model_g.id_a.load_state_dict(state_dict_id['a'])
        model_g.id_b = model_g.id_a
        
        model_g.to(device)
        model_g.eval()


    print("Loading data")
    train_loader = build_train_loader(cfg)
    gallery_loader, query_loader = build_test_loader(cfg)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        exit(0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        print("EPOCH : ", epoch)
        train_one_epoch(cfg, model, optimizer, train_loader, device, epoch, tfboard=None, model_g = model_g)
        lr_scheduler.step()

        if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                osp.join(output_dir, f"epoch_color{epoch}.pth"),
            )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    main(args)

