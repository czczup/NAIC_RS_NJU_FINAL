import os
import logging
import json
import torch
import subprocess
from .distributed import get_rank, synchronize
from .logger import setup_logger
from .env import seed_all_rng
from ..config import cfg
import torch.distributed as dist
import torch.multiprocessing as mp

def default_setup(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1

    if not args.no_cuda and torch.cuda.is_available():
        # cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"

    if args.launcher != 'none':
        init_dist(args.launcher, backend='nccl')

    save_dir = cfg.TRAIN.LOG_SAVE_DIR if cfg.PHASE == 'train' else None
    setup_logger("Segmentron", os.path.join(save_dir, cfg.DIR_NAME), get_rank(),
                 filename=args.config_file.split("/")[-1].split(".")[0]+"_"+str(cfg.TIME_STAMP)+'.txt')

    logging.info("Using {} GPUs".format(num_gpus))
    logging.info(args)
    logging.info(json.dumps(cfg, indent=8))

    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + get_rank())


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_slurm(backend, port=29500, **kwargs):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        'scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    
    
def _init_dist_pytorch_multi_nodes(backend, port=23410, **kwargs):
    def get_ip(string):
        items = string.replace("[", "").split("-")
        ip = ".".join(items[2:6])
        return ip
    
    ip = get_ip(os.environ['SLURM_STEP_NODELIST'])
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    host_addr_full = 'tcp://' + ip + ':' + str(port)
    print(host_addr_full, rank, local_rank, world_size)
    torch.distributed.init_process_group("nccl", init_method="env://")
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    assert torch.distributed.is_initialized()
    # dist.init_process_group(backend=backend, **kwargs)
