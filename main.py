import argparse
import os
import torch

import engine, logger
from dataset import CheXpertDatasetLoaderLite
from models import get_convnext_model

# ===
# DDP setup
# ===
from torch.distributed import init_process_group, destroy_process_group

is_ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

# torchrun cmd sets env vars: RANK, LOCAL_RANK, and WORLD_SIZE
if is_ddp: 
  assert torch.cuda.is_available(), "DDP requires CUDA"
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  print('DDP Rank:', ddp_rank)
  print('DDP Local Rank:', ddp_local_rank)
  print('DDP World Size:', ddp_world_size)
  DEVICE = f'cuda:{ddp_local_rank}'
  torch.cuda.set_device(DEVICE)
  is_master_process = ddp_rank == 0 # master process will do the logging, checkpointing, etc.
else:
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  DEVICE = 'cpu'
  if torch.cuda.is_available(): DEVICE = 'cuda'
  if torch.backends.mps.is_available(): DEVICE = 'mps'
  is_master_process = True


# ===
# Constants
# ===
argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=3e-4)
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--micro_batch_size', type=int, default=0)
argparser.add_argument('--n_steps', type=int, default=5000)
argparser.add_argument('--dropout_rate', type=float, default=0.1)
argparser.add_argument('--data_dir', type=str, default="/scratch/rawhad/CSE507/practice_1/chexpert_preprocessed_ds")
argparser.add_argument('--use_worker', action='store_true')
argparser.add_argument('--prefetch_size', type=int, default=1)
argparser.add_argument('--last_step', type=int, default=-1)
argparser.add_argument('--save_model', action='store_true')
argparser.add_argument('--do_overfit', action='store_true')
argparser.add_argument('--project_name', type=str, default='chexpert')
argparser.add_argument('--run_name', type=str, default='test')
argparser.add_argument('--model_name', type=str, default='convnext_tiny')
args = argparser.parse_args()

LR = args.lr
TOTAL_BATCH_SIZE = args.batch_size
MICRO_BATCH_SIZE = args.micro_batch_size if args.micro_batch_size > 0 else TOTAL_BATCH_SIZE
assert TOTAL_BATCH_SIZE % (MICRO_BATCH_SIZE * ddp_world_size) == 0, "Total batch size must be divisible by (micro batch size * world_size)"

GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // (MICRO_BATCH_SIZE * ddp_world_size)
N_STEPS = args.n_steps
DROPOUT_RATE = args.dropout_rate
USE_WORKER = args.use_worker
PREFETCH_SIZE = args.prefetch_size
DATA_DIR = args.data_dir
SAVE_MODEL = args.save_model
PROJECT_NAME = args.project_name
RUN_NAME = args.run_name
LAST_STEP = args.last_step
DO_OVERFIT = args.do_overfit
MODEL_NAME = args.model_name

MAX_STEPS = min(N_STEPS // 3, 2408) if DO_OVERFIT else 2408 # ~1epoch
WARMUP_STEPS = int(MAX_STEPS * 0.037) # based on build_nanogpt ratio
MAX_LR = LR
MIN_LR = LR / 10

MODEL_DIR = '/scratch/rawhad/CSE507/practice_1/models'

if is_master_process:
  os.makedirs(MODEL_DIR, exist_ok=True)
  LOGGER = logger.WandbLogger(project_name=PROJECT_NAME, run_name=RUN_NAME)
  #LOGGER = logger.ConsoleLogger(project_name='cse507_practice_1', run_name='test-chexpert')
  print('GRAD_ACCUM_STEPS: ', GRAD_ACCUM_STEPS)
else:
  LOGGER = None

# ===
# Intialization
# ===
torch.manual_seed(1234)  # setting seed because we are using DDP
if torch.cuda.is_available(): torch.cuda.manual_seed(1234)

train_ds = CheXpertDatasetLoaderLite(split='train', batch_size=MICRO_BATCH_SIZE, root=DATA_DIR, process_rank=ddp_rank, world_size=ddp_world_size, prefetch_size=PREFETCH_SIZE, use_worker=USE_WORKER)
test_ds = CheXpertDatasetLoaderLite(split='valid', batch_size=MICRO_BATCH_SIZE, root=DATA_DIR, process_rank=ddp_rank, world_size=ddp_world_size, prefetch_size=1)

model = get_convnext_model(MODEL_NAME, num_classes=14)

# ===
# Configure Optimizers and LR Schedulers
# ===

# lr scheduler
lr_scheduler = engine.CosineLRScheduler(WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)

training_config = engine.EngineConfig(
  train_ds=train_ds,
  test_ds=test_ds,
  model=model,
  N_STEPS=N_STEPS,
  device=DEVICE,
  logger=LOGGER,
  lr_scheduler=lr_scheduler,
  grad_accum_steps=GRAD_ACCUM_STEPS,
  checkpoint_every=1000,
  checkpoint_dir=MODEL_DIR,
  last_step=LAST_STEP,
  is_ddp=is_ddp,
  ddp_rank=ddp_rank,
  ddp_local_rank=ddp_local_rank,
  ddp_world_size=ddp_world_size,
  is_master_process=is_master_process,
  do_overfit=DO_OVERFIT,
)
engine.run(training_config)

if is_master_process:
  # save models
  if SAVE_MODEL:
    os.makedirs(os.path.join(MODEL_DIR, RUN_NAME), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, RUN_NAME, 'model_final.pth'))

if is_ddp:
  destroy_process_group()
