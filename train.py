import argparse
import torch
import random
import numpy as np

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from utils.utils import seed_everything
# from dassl.config import get_cfg_default
from utils import cfg_default
from dassl.engine import build_trainer
from trainer import Trainer
from bitfit_trainer import BitFitTrainer
from ln_only_trainer import LayerNormOnlyTrainer

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root


    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.output_dir:
        if cfg.TRAINER.NAME in ['BitFitTrainer', 'LayerNormOnlyTrainer']:
            peft_method = cfg.TRAINER.NAME.replace('Trainer', '').lower()
            cfg.OUTPUT_DIR = f'{args.output_dir}/{cfg.MODEL.NAME}_{peft_method}/{"_".join(cfg.DATASET.SOURCE_DOMAINS)}/{"_".join(cfg.DATASET.TARGET_DOMAINS)}'
        else:
            cfg.OUTPUT_DIR = f'{args.output_dir}/{cfg.MODEL.NAME}/{"_".join(cfg.DATASET.SOURCE_DOMAINS)}/{"_".join(cfg.DATASET.TARGET_DOMAINS)}'

    if args.use_align_loss:
        cfg.TRAINER.USE_ALIGN_LOSS = args.use_align_loss
    
    if args.use_sinkhorn_loss:
        cfg.TRAINER.USE_SINKHORN_LOSS = args.use_sinkhorn_loss
    
    if args.use_prototype_loss:
        cfg.TRAINER.USE_PROTOTYPE_LOSS = args.use_prototype_loss
    
    if args.use_kl_loss:
        cfg.TRAINER.USE_KL_LOSS = args.use_kl_loss
    
    if args.use_w1_loss:
        cfg.TRAINER.USE_W1_LOSS = args.use_w1_loss
    
    if args.use_entropy_loss:
        cfg.TRAINER.USE_ENTROPY_LOSS = args.use_entropy_loss
    

    if args.use_confidence_sampling:
        cfg.TRAINER.USE_CONFIDENCE_SAMPLING = args.use_confidence_sampling



def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.NAME = 'Trainer'
    cfg.TRAINER.MODEL = CN()
    cfg.TRAINER.MODEL.N_CTX = 4  # number of context vectors
    cfg.TRAINER.MODEL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.MODEL.PREC = "fp32"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = cfg_default.clone()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg
    
def reproducible_setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        reproducible_setup(cfg.SEED)    
        # set_random_seed(cfg.SEED)
        # seed_everything(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="PointDA_data", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="test_runs_with_sinkhorn", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/trainer.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/pointda_shapenet_modelnet.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    parser.add_argument("--use_align_loss", action="store_true", help="Use alignment loss")
    parser.add_argument("--use_sinkhorn_loss", action="store_true", help="Use Sinkhorn loss")
    parser.add_argument("--use_prototype_loss", action="store_true", help="Use Prototype loss")
    parser.add_argument("--use_kl_loss", action="store_true", help="Use KL divergence loss")
    parser.add_argument("--use_w1_loss", action="store_true", help="Use Wasserstein-1 loss")
    parser.add_argument("--use_entropy_loss", action="store_true", help="Use entropy loss")

    parser.add_argument("--use_confidence_sampling", action="store_true", help="Use confidence sampling")


    args = parser.parse_args()
    main(args)