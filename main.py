# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import yaml
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch

from models import build_model as build_yolos_model

from util.scheduler import create_scheduler


def get_args_parser():
    parser = argparse.ArgumentParser('Set YOLOS', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--backbone-lr', default=1e-5, type=float)
    parser.add_argument('--layer-decay', default=1, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help='use checkpoint.checkpoint to save mem')
    parser.add_argument('--evaluator', default='coco', type=str)
    parser.add_argument('--eval-size', default=800, type=int)
    parser.add_argument('--eval-during-training', action='store_true')
    parser.add_argument('--eval-epochs', default=1, type=int)
    # scheduler
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='warmupcos', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step", options:"step", "warmupcos"')
    ## step
    parser.add_argument('--lr-drop', default=100, type=int)  
    ## warmupcosine

    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * model setting
    parser.add_argument('--num-det-token', default=100, type=int,
                        help="Number of det token in the deit backbone")
    parser.add_argument('--backbone', default='dinov3', type=str,
                        help="Name of the backbone to use"),
    parser.add_argument('--backbone-size', default='dinov3', type=str,
                        help="Size of the backbone to use"),
    parser.add_argument('--unfreeze', nargs='+', default=[]),
    parser.add_argument('--no-weight-decay', nargs='+', default=[]),
    parser.add_argument('--pretrained-path', default= None),
    parser.add_argument('--init-pe-size', nargs='+', type=int,
                        help="init pe size (h,w)")
    parser.add_argument('--mid-pe-size', nargs='+', type=int,
                        help="mid pe size (h,w)")
    # * Matcher
    parser.add_argument('--set-cost-class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set-cost-bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set-cost-giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--ce-loss-coef', default=1, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset-file', default='coco')
    parser.add_argument('--coco-path', type=str)
    parser.add_argument('--remove-difficult', action='store_true')

    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save-epochs', default=10, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--bf16', action='store_true',
                        help='Enable bfloat16 mixed precision training')

    parser.add_argument('--config-path',default=None, help='path to .yaml configuration file')
    return parser


def load_config(path, args, cli_args):
    if path is None:
        return args
    
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    
    for k, v in cfg.items():
        if v is not None and k not in cli_args:
            setattr(args, k, v)
    
    return args


def main(args):
    # print("git:\n  {}\n".format(utils.get_sha()))
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # import pdb;pdb.set_trace()
    model, criterion, postprocessors = build_yolos_model(args)
    # model, criterion, postprocessors = build_model(args)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    def build_optimizer(model, args):
        skip = args.no_weight_decay
        head = []
        backbone_decay = []
        backbone_no_decay = []
        for name, param in model.named_parameters():
            if "backbone" not in name and param.requires_grad:
                head.append(param)
            if "backbone" in name and param.requires_grad:
                if len(param.shape) == 1 or name.endswith(".bias") or name.split('.')[-1] in skip:
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)
        param_dicts = [
            {"params": head},
            {"params": backbone_no_decay, "weight_decay": 0., "lr": args.backbone_lr},
            {"params": backbone_decay, "lr": args.backbone_lr},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
        return optimizer

    optimizer = build_optimizer(model, args)


    lr_scheduler, _ = create_scheduler(args, optimizer)
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_base = build_dataset(image_set='base', args=args)

    # import pdb;pdb.set_trace()
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_base)


    output_dir = Path(args.output_dir)
    tb_writer = None
    if args.output_dir and utils.is_main_process():
        (output_dir / 'tensorboard').mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, evaluator_obj = evaluate(evaluator=args.evaluator, model=model, criterion=criterion, postprocessors=postprocessors, data_loader=data_loader_val, base_ds=base_ds, device=device, epoch=args.start_epoch-1, tb_writer=tb_writer)
        if args.output_dir and args.evaluator == 'coco' and evaluator_obj is not None:
            utils.save_on_master(evaluator_obj.coco_eval["bbox"].eval, output_dir / "eval.pth")
        if tb_writer is not None:
            tb_writer.close()
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, tb_writer=tb_writer, use_bf16=args.bf16)
        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_epochs == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters}
        
        do_eval = args.eval_during_training and ((epoch + 1) % args.eval_epochs == 0 or (epoch + 1) == args.epochs)
        if do_eval:
            test_stats, evaluator_obj = evaluate(evaluator=args.evaluator, model=model, criterion=criterion, postprocessors=postprocessors, data_loader=data_loader_val, base_ds=base_ds, device=device, epoch=epoch, tb_writer=tb_writer)
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            # for COCO evaluation logs
            if args.evaluator == 'coco' and evaluator_obj is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in evaluator_obj.coco_eval:
                    filenames = ['latest.pth']
                    filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(evaluator_obj.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOS training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    cli_args = set(arg.lstrip('-').replace('-', '_') 
                   for arg in sys.argv[1:] 
                   if arg.startswith('-') and not arg.startswith('--config_path'))
    
    args = load_config(args.config_path, args, cli_args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)