#!/usr/bin/env python3
"""
train_discrete.py - 离散化 phi 的训练脚本

将 phi 角度离散化为 36 个 bin，使用 CrossEntropy 损失
解决多模态动作的"平均化"问题

使用方法:
    python train_discrete.py --data_dir ./data_100k --output_dir ./checkpoints_discrete
"""

import os
import sys
import subprocess

# ============================================================
# GPU 选择 - 必须在 import torch 之前完成
# ============================================================
def _select_gpu_before_torch():
    """在 import torch 之前选择 GPU 并设置环境变量"""
    # 检查命令行参数中的 --device
    device_arg = None
    for i, arg in enumerate(sys.argv):
        if arg == '--device' and i + 1 < len(sys.argv):
            device_arg = sys.argv[i + 1]
            break
    
    # 如果没有指定或指定为 auto，自动选择
    if device_arg is None or device_arg == 'auto':
        gpu_id = _get_free_gpu_nvidia_smi()
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"[GPU] Auto-selected physical GPU {gpu_id}")
            print(f"[GPU] Set CUDA_VISIBLE_DEVICES={gpu_id}")
    elif device_arg.startswith('cuda:'):
        # 用户指定了特定 GPU
        gpu_id = device_arg.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f"[GPU] Using specified GPU {gpu_id}")


def _get_free_gpu_nvidia_smi():
    """使用 nvidia-smi 获取最空闲的 GPU (不依赖 torch)"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode != 0:
            return 0
        
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                idx = int(parts[0])
                mem_used = float(parts[1])
                mem_total = float(parts[2])
                util = float(parts[3])
                
                mem_free_ratio = 1 - (mem_used / mem_total)
                util_free_ratio = 1 - (util / 100)
                score = mem_free_ratio * 0.7 + util_free_ratio * 0.3
                
                gpu_info.append((idx, score, mem_used, mem_total, util))
        
        if not gpu_info:
            return 0
        
        gpu_info.sort(key=lambda x: x[1], reverse=True)
        best_gpu = gpu_info[0]
        
        print(f"[GPU Selection] Available GPUs:")
        for idx, score, mem_used, mem_total, util in gpu_info:
            marker = " ← selected" if idx == best_gpu[0] else ""
            print(f"  GPU {idx}: {mem_used:.0f}/{mem_total:.0f} MB ({util:.0f}% util){marker}")
        
        return best_gpu[0]
        
    except Exception as e:
        print(f"[GPU Selection] Warning: {e}, defaulting to GPU 0")
        return 0


# 在 import torch 之前执行 GPU 选择
_select_gpu_before_torch()

# ============================================================
# 现在可以安全地 import torch
# ============================================================
import argparse
import glob
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from model_discrete import (
    create_discrete_model, 
    sincos_to_bin, 
    bin_to_angle,
    angle_to_bin,
)


class BilliardsDiscreteDataset(Dataset):
    """
    离散化 phi 的数据集
    
    原始 action: [V0, phi_sin, phi_cos, theta, a, b]
    转换为: 
        - continuous: [V0, theta, a, b]
        - phi_bin: int (0-35)
    """
    
    def __init__(self, states, actions, num_phi_bins=36, augment=True, noise_std=0.0):
        self.num_phi_bins = num_phi_bins
        self.noise_std = noise_std
        
        if augment:
            states, actions = self._augment_symmetric(states, actions)
        
        self.states = torch.from_numpy(states).float()
        
        # 分离 phi 和其他连续动作
        # 原始 actions: [V0, phi_sin, phi_cos, theta, a, b]
        continuous_actions = np.zeros((len(actions), 4), dtype=np.float32)
        continuous_actions[:, 0] = actions[:, 0]  # V0
        continuous_actions[:, 1] = actions[:, 3]  # theta
        continuous_actions[:, 2] = actions[:, 4]  # a
        continuous_actions[:, 3] = actions[:, 5]  # b
        
        self.continuous_actions = torch.from_numpy(continuous_actions).float()
        
        # 将 phi sin/cos 转换为 bin
        phi_bins = np.zeros(len(actions), dtype=np.int64)
        for i in range(len(actions)):
            phi_sin = actions[i, 1]
            phi_cos = actions[i, 2]
            phi_bins[i] = sincos_to_bin(phi_sin, phi_cos, num_phi_bins)
        
        self.phi_bins = torch.from_numpy(phi_bins).long()
        
        print(f"[Dataset] Samples: {len(self.states)}, Phi bins: {num_phi_bins}")
        
        # 统计 phi 分布
        bin_counts = np.bincount(phi_bins, minlength=num_phi_bins)
        print(f"[Dataset] Phi distribution: min={bin_counts.min()}, max={bin_counts.max()}, "
              f"mean={bin_counts.mean():.1f}")
    
    def _augment_symmetric(self, states, actions):
        """对称性增强 (4倍)"""
        n = len(states)
        
        aug_states = np.zeros((n * 4, states.shape[1]), dtype=np.float32)
        aug_actions = np.zeros((n * 4, actions.shape[1]), dtype=np.float32)
        
        # 原始
        aug_states[0:n] = states
        aug_actions[0:n] = actions
        
        # 左右镜像
        aug_states[n:2*n] = self._mirror_lr_state(states)
        aug_actions[n:2*n] = self._mirror_lr_action(actions)
        
        # 上下镜像
        aug_states[2*n:3*n] = self._mirror_ud_state(states)
        aug_actions[2*n:3*n] = self._mirror_ud_action(actions)
        
        # 180度旋转
        aug_states[3*n:4*n] = self._rotate_180_state(states)
        aug_actions[3*n:4*n] = self._rotate_180_action(actions)
        
        return aug_states, aug_actions
    
    def _mirror_lr_state(self, states):
        """左右镜像状态"""
        s = states.copy()
        s[:, 1] = 1.0 - s[:, 1]  # 白球 y
        for i in range(15):
            s[:, 3 + i * 3 + 1] = 1.0 - s[:, 3 + i * 3 + 1]  # 球 y
        # 袋口交换和镜像 (简化处理)
        s_pockets = s[:, 63:75].copy()
        s[:, 64] = 1.0 - s_pockets[:, 1]
        s[:, 66] = 1.0 - s_pockets[:, 3]
        s[:, 68] = 1.0 - s_pockets[:, 5]
        s[:, 70] = 1.0 - s_pockets[:, 7]
        s[:, 72] = 1.0 - s_pockets[:, 9]
        s[:, 74] = 1.0 - s_pockets[:, 11]
        return s
    
    def _mirror_lr_action(self, actions):
        """左右镜像动作 (phi -> -phi)"""
        a = actions.copy()
        a[:, 1] = -a[:, 1]  # phi_sin 取反
        return a
    
    def _mirror_ud_state(self, states):
        """上下镜像状态"""
        s = states.copy()
        s[:, 0] = 1.0 - s[:, 0]  # 白球 x
        for i in range(15):
            s[:, 3 + i * 3] = 1.0 - s[:, 3 + i * 3]  # 球 x
        # 袋口 x 镜像
        s_pockets = s[:, 63:75].copy()
        s[:, 63] = 1.0 - s_pockets[:, 0]
        s[:, 65] = 1.0 - s_pockets[:, 2]
        s[:, 67] = 1.0 - s_pockets[:, 4]
        s[:, 69] = 1.0 - s_pockets[:, 6]
        s[:, 71] = 1.0 - s_pockets[:, 8]
        s[:, 73] = 1.0 - s_pockets[:, 10]
        return s
    
    def _mirror_ud_action(self, actions):
        """上下镜像动作 (phi -> 180-phi)"""
        a = actions.copy()
        a[:, 2] = -a[:, 2]  # phi_cos 取反
        return a
    
    def _rotate_180_state(self, states):
        """180度旋转"""
        s = states.copy()
        s[:, 0] = 1.0 - s[:, 0]  # 白球 x
        s[:, 1] = 1.0 - s[:, 1]  # 白球 y
        for i in range(15):
            s[:, 3 + i * 3] = 1.0 - s[:, 3 + i * 3]      # x
            s[:, 3 + i * 3 + 1] = 1.0 - s[:, 3 + i * 3 + 1]  # y
        # 袋口
        for j in range(6):
            s[:, 63 + j * 2] = 1.0 - s[:, 63 + j * 2]
            s[:, 63 + j * 2 + 1] = 1.0 - s[:, 63 + j * 2 + 1]
        return s
    
    def _rotate_180_action(self, actions):
        """180度旋转动作 (phi -> phi+180)"""
        a = actions.copy()
        a[:, 1] = -a[:, 1]  # phi_sin
        a[:, 2] = -a[:, 2]  # phi_cos
        return a
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        
        if self.noise_std > 0:
            noise = torch.randn_like(state) * self.noise_std
            noise_mask = torch.ones_like(state)
            noise_mask[2] = 0
            for i in range(15):
                noise_mask[3 + i * 3 + 2] = 0
            noise_mask[75:80] = 0
            state = state + noise * noise_mask
        
        return state, self.continuous_actions[idx], self.phi_bins[idx]


class DiscreteLoss(nn.Module):
    """
    离散化 phi 的损失函数
    
    - phi: CrossEntropy with label smoothing
    - V0, theta, spin: MSE
    """
    
    def __init__(self, num_phi_bins=36, label_smoothing=0.1, weights=None):
        super().__init__()
        
        self.num_phi_bins = num_phi_bins
        self.weights = weights or {
            'v0': 1.0,
            'phi': 2.0,  # 角度仍然重要
            'theta': 1.0,
            'spin': 0.5,
        }
        
        # 带 label smoothing 的交叉熵
        self.phi_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mse = nn.MSELoss()
    
    def forward(self, output, continuous_target, phi_target):
        """
        参数:
            output: dict with 'v0', 'phi_logits', 'theta', 'spin'
            continuous_target: [B, 4] - [V0, theta, a, b]
            phi_target: [B] - bin indices
        """
        # V0 损失
        v0_loss = self.mse(output['v0'][:, 0], continuous_target[:, 0])
        
        # Phi 损失 (CrossEntropy)
        phi_loss = self.phi_criterion(output['phi_logits'], phi_target)
        
        # Theta 损失
        theta_loss = self.mse(output['theta'][:, 0], continuous_target[:, 1])
        
        # Spin 损失
        spin_loss = self.mse(output['spin'], continuous_target[:, 2:4])
        
        # 总损失
        total_loss = (
            self.weights['v0'] * v0_loss +
            self.weights['phi'] * phi_loss +
            self.weights['theta'] * theta_loss +
            self.weights['spin'] * spin_loss
        )
        
        # 计算 phi 准确率
        with torch.no_grad():
            phi_pred = output['phi_logits'].argmax(dim=1)
            phi_acc = (phi_pred == phi_target).float().mean()
            
            # 允许 ±1 bin 误差的准确率 (考虑循环)
            diff = torch.abs(phi_pred - phi_target)
            diff = torch.min(diff, self.num_phi_bins - diff)
            phi_acc_relaxed = (diff <= 1).float().mean()
        
        return total_loss, {
            'v0': v0_loss.item(),
            'phi_ce': phi_loss.item(),
            'theta': theta_loss.item(),
            'spin': spin_loss.item(),
            'phi_acc': phi_acc.item(),
            'phi_acc_relaxed': phi_acc_relaxed.item(),
        }


class DiscreteTrainer:
    """离散 phi 训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        output_dir,
        use_amp=True,
        log_interval=100,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.log_interval = log_interval
        
        self.scaler = GradScaler() if use_amp else None
        self.writer = SummaryWriter(os.path.join(output_dir, 'logs'))
        
        self.best_val_loss = float('inf')
        self.best_phi_acc = 0.0
        self.patience_counter = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        self.model.train()
        
        total_loss = 0.0
        metrics = {'v0': 0, 'phi_ce': 0, 'theta': 0, 'spin': 0, 'phi_acc': 0, 'phi_acc_relaxed': 0}
        num_batches = 0
        
        for batch_idx, (states, continuous, phi_bins) in enumerate(self.train_loader):
            states = states.to(self.device)
            continuous = continuous.to(self.device)
            phi_bins = phi_bins.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(states)
                    loss, losses = self.criterion(output, continuous, phi_bins)
                
                # 检查 loss 是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    continue  # 跳过这个 batch
                
                self.scaler.scale(loss).backward()
                
                # 检查梯度是否有效
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                # 如果梯度异常，跳过更新
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"\n[WARNING] NaN/Inf gradient at batch {batch_idx}! Skipping update.")
                    self.optimizer.zero_grad()
                    continue
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(states)
                loss, losses = self.criterion(output, continuous, phi_bins)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"\n[WARNING] NaN/Inf gradient at batch {batch_idx}! Skipping update.")
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()
            
            # NaN 检测
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] NaN/Inf detected at batch {batch_idx}! Skipping batch.")
                self.nan_count = getattr(self, 'nan_count', 0) + 1
                if self.nan_count > 10:
                    raise RuntimeError("Too many NaN losses! Training stopped.")
                continue
            
            total_loss += loss.item()
            for k in metrics:
                metrics[k] += losses[k]
            num_batches += 1
            
            if batch_idx % self.log_interval == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | Phi Acc: {losses['phi_acc']:.2%}")
        
        avg_loss = total_loss / num_batches
        for k in metrics:
            metrics[k] /= num_batches
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        
        total_loss = 0.0
        metrics = {'v0': 0, 'phi_ce': 0, 'theta': 0, 'spin': 0, 'phi_acc': 0, 'phi_acc_relaxed': 0}
        num_batches = 0
        
        for states, continuous, phi_bins in self.val_loader:
            states = states.to(self.device)
            continuous = continuous.to(self.device)
            phi_bins = phi_bins.to(self.device)
            
            if self.use_amp:
                with autocast():
                    output = self.model(states)
                    loss, losses = self.criterion(output, continuous, phi_bins)
            else:
                output = self.model(states)
                loss, losses = self.criterion(output, continuous, phi_bins)
            
            total_loss += loss.item()
            for k in metrics:
                metrics[k] += losses[k]
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        for k in metrics:
            metrics[k] /= num_batches
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, val_loss, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'phi_acc': metrics['phi_acc'],
            'model_config': {
                'state_dim': self.model.state_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_phi_bins': self.model.num_phi_bins,
            }
        }
        
        torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_latest.pt'))
        
        if epoch % 10 == 0:
            torch.save(checkpoint, os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_best.pt'))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model_best.pt'))
    
    def train(self, num_epochs, patience=20, start_epoch=1):
        print(f"\n[INFO] Starting discrete phi training for {num_epochs} epochs")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Phi bins: {self.model.num_phi_bins}")
        
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            start_time = time.time()
            train_loss, train_metrics = self.train_epoch(epoch)
            train_time = time.time() - start_time
            
            val_loss, val_metrics = self.validate(epoch)
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  V0: {val_metrics['v0']:.4f} | Phi CE: {val_metrics['phi_ce']:.4f} | "
                  f"Theta: {val_metrics['theta']:.4f} | Spin: {val_metrics['spin']:.4f}")
            print(f"  Phi Acc: {val_metrics['phi_acc']:.2%} | Phi Acc (±1 bin): {val_metrics['phi_acc_relaxed']:.2%}")
            print(f"LR: {current_lr:.6f} | Time: {train_time:.1f}s")
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/phi', val_metrics['phi_acc'], epoch)
            self.writer.add_scalar('Accuracy/phi_relaxed', val_metrics['phi_acc_relaxed'], epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # 使用 val_loss 判断最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_phi_acc = val_metrics['phi_acc']
                self.patience_counter = 0
                print(f"  ★ New best model! Phi Acc: {val_metrics['phi_acc']:.2%}")
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, val_metrics, is_best)
            
            if self.patience_counter >= patience:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break
        
        self.writer.close()
        print(f"\n[INFO] Training completed.")
        print(f"[INFO] Best val loss: {self.best_val_loss:.4f}, Best phi acc: {self.best_phi_acc:.2%}")
        
        return self.best_val_loss


def load_data(data_dir):
    """加载数据"""
    state_files = sorted(glob.glob(os.path.join(data_dir, 'states_*.npy')))
    action_files = sorted(glob.glob(os.path.join(data_dir, 'actions_*.npy')))
    
    if not state_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    print(f"[INFO] Found {len(state_files)} data files")
    
    all_states = []
    all_actions = []
    
    for sf, af in zip(state_files, action_files):
        states = np.load(sf)
        actions = np.load(af)
        all_states.append(states)
        all_actions.append(actions)
    
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    print(f"[INFO] Total samples: {len(states)}")
    
    return states, actions


def main():
    parser = argparse.ArgumentParser(description='Train with discrete phi')
    
    parser.add_argument('--data_dir', type=str, default='./data_100k')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_discrete')
    
    parser.add_argument('--model_type', type=str, default='large', choices=['large', 'small'])
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_phi_bins', type=int, default=36, help='Number of phi bins (default: 36 = 10° each)')
    
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)  # 降低默认LR防止NaN
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto (select free GPU), cuda, cuda:0, cuda:1, cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--no_amp', action='store_true')
    
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # 设备选择 (环境变量已在 import torch 之前设置)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # 始终用 cuda:0，因为 CUDA_VISIBLE_DEVICES 已限制
    else:
        device = torch.device('cpu')
    
    print(f"[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # 加载数据
    states, actions = load_data(args.data_dir)
    
    # 划分 train/val
    n = len(states)
    val_size = int(n * args.val_split)
    train_size = n - val_size
    
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_states = states[train_indices]
    train_actions = actions[train_indices]
    val_states = states[val_indices]
    val_actions = actions[val_indices]
    
    print(f"[INFO] Train: {len(train_states)}, Val: {len(val_states)}")
    
    # 创建数据集
    train_dataset = BilliardsDiscreteDataset(
        train_states, train_actions,
        num_phi_bins=args.num_phi_bins,
        augment=not args.no_augment,
        noise_std=args.noise_std,
    )
    
    val_dataset = BilliardsDiscreteDataset(
        val_states, val_actions,
        num_phi_bins=args.num_phi_bins,
        augment=False,
        noise_std=0.0,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # 创建模型
    model_kwargs = {
        'state_dim': states.shape[1],
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'num_phi_bins': args.num_phi_bins,
    }
    
    if args.model_type == 'large':
        model_kwargs['num_heads'] = args.num_heads
    
    model = create_discrete_model(args.model_type, **model_kwargs)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {num_params:,}")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6,
    )
    
    # 损失函数
    criterion = DiscreteLoss(
        num_phi_bins=args.num_phi_bins,
        label_smoothing=args.label_smoothing,
    )
    
    # 恢复检查点
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f"[INFO] Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO] Resumed from epoch {checkpoint['epoch']}")
    
    # 训练
    trainer = DiscreteTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        output_dir=args.output_dir,
        use_amp=not args.no_amp,
    )
    
    trainer.train(args.num_epochs, args.patience, start_epoch=start_epoch)


if __name__ == '__main__':
    main()

