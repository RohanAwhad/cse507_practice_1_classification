# engine.py
import dataclasses
import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score

from dataset import Dataset
from logger import Logger

# ===
# Configure Optimizer
# ===
def configure_optimizer(model, weight_decay, lr, device_type):
  # get all parameters that require grad
  params = filter(lambda p: p.requires_grad, model.parameters())
  # create param groups based on ndim
  optim_groups = [
    {'params': [p for p in params if p.ndim >= 2], 'weight_decay': weight_decay},
    {'params': [p for p in params if p.ndim < 1], 'weight_decay': 0.0},
  ]
  # use fused optimizer if available
  fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
  use_fused = fused_available and device_type == 'cuda'
  return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)


# ===
# LR Scheduler
# ===
class CosineLRScheduler:
  def __init__(self, warmup_steps, max_steps, max_lr, min_lr):
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps
    self.max_lr = max_lr
    self.min_lr = min_lr

  def get_lr(self, step):
    # linear warmup
    if step < self.warmup_steps:
      return self.max_lr * (step+1) / self.warmup_steps

    # constant lr
    if step > self.max_steps:
      return self.min_lr

    # cosine annealing
    decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return self.min_lr + coeff * (self.max_lr - self.min_lr)



# ===
# Engine
# ===
@dataclasses.dataclass
class EngineConfig:
  train_ds: Dataset
  test_ds: Dataset
  model: nn.Module
  N_STEPS: int
  device: str
  logger: Logger
  lr_scheduler: CosineLRScheduler
  grad_accum_steps: int
  checkpoint_every: int
  checkpoint_dir: str
  last_step: int
  # ddp vars
  is_ddp: bool
  ddp_rank: int
  ddp_local_rank: int
  ddp_world_size: int
  is_master_process: bool
  do_overfit: bool


def run(config: EngineConfig):
  PLOT_EVERY = 100 if config.do_overfit else 1000
  EVAL_EVERY = 1000

  device_type = 'cuda' if config.device.startswith('cuda') else config.device

  config.model.to(config.device)

  if config.last_step != -1:
    print("Resuming training from step:", config.last_step)
    # load model and optimizer state
    config.model.load_state_dict(torch.load(f'{config.checkpoint_dir}/model.pth'))
    with open(f'{config.checkpoint_dir}/last_step.txt', 'r') as f: start_step = int(f.read())
  else:
    if config.is_master_process: print("Starting training from scratch")
    start_step = 0

  raw_model = config.model
  if config.is_ddp:
    config.model = nn.parallel.DistributedDataParallel(config.model, device_ids=[config.ddp_local_rank])

  model_opt = configure_optimizer(raw_model, 0.1, config.lr_scheduler.max_lr, device_type) # lr is placeholder
  model_opt.zero_grad()

  if config.last_step != -1:
    # load optimizer state
    model_opt.load_state_dict(torch.load(f'{config.checkpoint_dir}/model_opt.pth'))

  if config.do_overfit:
    test_samples = config.train_ds.next_batch()
    test_images = test_samples['image'].to(config.device)
  else:
    test_samples = config.test_ds.next_batch()
    test_images = test_samples['image'].to(config.device)

  if config.is_master_process: print("Training for steps:", config.N_STEPS)
  config.model.train()

  for step in range(start_step, config.N_STEPS):
    if config.do_overfit: config.train_ds.reset()
    start = time.monotonic()
    log_data = dict(
      model=dict(total_loss = 0.0),
    )

    config.model.train()
    model_opt.zero_grad()
    for micro_step in range(config.grad_accum_steps):
      batch = config.train_ds.next_batch()
      images = batch['image'].to(config.device)
      labels = batch['label'].to(config.device)
      label_masks = batch['label_mask'].to(config.device)

      with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        outputs = config.model(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
        loss = (loss * label_masks).sum() / label_masks.sum()
        loss = loss / config.grad_accum_steps
      loss.backward()

      # log
      log_data['model']['total_loss'] += loss.detach()

    lr = config.lr_scheduler.get_lr(step)
    for param_group in model_opt.param_groups: param_group['lr'] = lr

    model_opt.step()

    log_data['lr'] = lr
    if config.is_ddp:
      dist.all_reduce(log_data['model']['total_loss'], op=dist.ReduceOp.AVG)
    if config.is_master_process: config.logger.log(log_data, step=step)

    end = time.monotonic()
    if config.is_master_process: print(f'Time taken for step {step}: {end-start:0.2f} secs')

    # periodically plot test images
    if step % PLOT_EVERY == 0 and config.is_master_process:

      with torch.no_grad():
        config.model.eval()
        outputs = config.model(test_images)
        probs = torch.sigmoid(outputs)
        config.model.train()
      
      # calculate loss
      loss = F.binary_cross_entropy_with_logits(outputs, test_samples['label'].to(config.device))
      print('Test Loss:', loss.item())

      # visualize
      fig, axs = plt.subplots(1, 2, figsize=(12, 6))
      axs[0].imshow(test_images[0].permute(1, 2, 0).detach().cpu().numpy())
      axs[0].axis('off')
      axs[0].set_title('Original Image')
      axs[1].bar(range(14), probs[0].detach().cpu().numpy())
      axs[1].set_title('Predicted Probabilities')
      config.logger.log(dict(test_images = fig), step=step)
      plt.close()

    # periodically evaluate model
    if step % EVAL_EVERY == 0 and config.is_master_process:
      with torch.no_grad():
        config.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_probs = []
        for _ in range(2): # evaluate on 2 batches
          batch = config.test_ds.next_batch()
          images = batch['image'].to(config.device)
          labels = batch['label'].to(config.device)
          label_masks = batch['label_mask'].to(config.device)
          outputs = config.model(images)
          loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
          loss = (loss * label_masks).sum() / label_masks.sum()
          total_loss += loss.item()
          probs = torch.sigmoid(outputs)
          predicted = (probs > 0.5).int()
          total_correct += (predicted == labels).sum().item()
          total_samples += labels.shape[0] * labels.shape[1]
          all_labels.extend(labels.cpu().numpy())
          all_probs.extend(probs.cpu().numpy())
        avg_loss = total_loss / 2
        accuracy = total_correct / total_samples
        
        # Calculate AUC for each class separately
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        num_classes = all_labels.shape[1]
        aucs = []
        for i in range(num_classes):
          try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            aucs.append(auc)
          except ValueError:
            # Handle the case where only one class is present
            aucs.append(np.nan)
        
        # Calculate the average AUC
        avg_auc = np.nanmean(aucs)
        
        print(f'Evaluation at step {step}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}, AUC = {avg_auc:.4f}')
        config.logger.log(dict(evaluation_loss = avg_loss, evaluation_accuracy = accuracy, evaluation_auc = avg_auc), step=step)
        config.model.train()
