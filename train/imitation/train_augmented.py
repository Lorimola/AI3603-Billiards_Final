#!/usr/bin/env python3
"""
train_augmented.py - 带对称性增强的模仿学习训练脚本

改进点：
1. 对称性数据增强（4倍数据）
2. 可选的离散化/MDN损失
3. 状态噪声注入

使用方法:
    python train_augmented.py --data_dir ./data_100k --output_dir ./checkpoints_aug
"""

import os
import sys
import argparse
import glob
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from model import create_model


class BilliardsAugmentedDataset(Dataset):
    """
    带对称性增强的台球数据集
    
    状态特征结构 (80维):
    - [0:3]: 白球 (x, y, pocketed)
    - [3:48]: 15个球 (x, y, pocketed) * 15
    - [48:63]: 目标球mask (15)
    - [63:75]: 6个袋口 (x, y) * 6
    - [75:80]: 统计特征 (5)
    
    动作特征结构 (6维):
    - [0]: V0_norm
    - [1]: phi_sin
    - [2]: phi_cos
    - [3]: theta_norm
    - [4]: a_norm
    - [5]: b_norm
    """
    
    # 袋口映射: lb=左下, lc=左中, lt=左上, rb=右下, rc=右中, rt=右上
    # 索引在特征中: lb=63:65, lc=65:67, lt=67:69, rb=69:71, rc=71:73, rt=73:75
    
    def __init__(self, states, actions, augment=True, noise_std=0.0):
        """
        参数:
            states: np.ndarray [N, 80]
            actions: np.ndarray [N, 6]
            augment: 是否启用对称性增强
            noise_std: 状态噪声标准差
        """
        self.augment = augment
        self.noise_std = noise_std
        
        if augment:
            # 生成增强数据
            aug_states, aug_actions = self._generate_augmented_data(states, actions)
            self.states = torch.from_numpy(aug_states).float()
            self.actions = torch.from_numpy(aug_actions).float()
        else:
            self.states = torch.from_numpy(states).float()
            self.actions = torch.from_numpy(actions).float()
        
        print(f"[Dataset] Samples: {len(self.states)} (augment={augment})")
    
    def _generate_augmented_data(self, states, actions):
        """生成对称增强数据 (4倍)"""
        n = len(states)
        
        # 预分配内存
        aug_states = np.zeros((n * 4, states.shape[1]), dtype=np.float32)
        aug_actions = np.zeros((n * 4, actions.shape[1]), dtype=np.float32)
        
        # 原始数据
        aug_states[0:n] = states
        aug_actions[0:n] = actions
        
        # 左右镜像 (y -> 1-y, phi -> -phi)
        aug_states[n:2*n] = self._mirror_lr(states)
        aug_actions[n:2*n] = self._mirror_lr_action(actions)
        
        # 上下镜像 (x -> 1-x, phi -> 180-phi)
        aug_states[2*n:3*n] = self._mirror_ud(states)
        aug_actions[2*n:3*n] = self._mirror_ud_action(actions)
        
        # 180度旋转 (x->1-x, y->1-y, phi -> phi+180)
        aug_states[3*n:4*n] = self._rotate_180(states)
        aug_actions[3*n:4*n] = self._rotate_180_action(actions)
        
        return aug_states, aug_actions
    
    def _mirror_lr(self, states):
        """左右镜像状态 (y -> 1-y)"""
        s = states.copy()
        
        # 白球 y
        s[:, 1] = 1.0 - s[:, 1]
        
        # 15个球 y (索引1, 4, 7, ... 即 3+3i+1 for i in 0..14)
        for i in range(15):
            idx = 3 + i * 3 + 1  # y坐标索引
            s[:, idx] = 1.0 - s[:, idx]
        
        # 袋口 y - 需要交换左右袋口
        # lb(63:65) <-> rb(69:71), lc(65:67) <-> rc(71:73), lt(67:69) <-> rt(73:75)
        s_pockets = s[:, 63:75].copy()
        # 交换并镜像y
        s[:, 63:65] = s_pockets[:, 6:8]  # lb <- rb
        s[:, 63] = s_pockets[:, 6]       # x不变
        s[:, 64] = 1.0 - s_pockets[:, 7] # y镜像
        
        s[:, 65:67] = s_pockets[:, 8:10]  # lc <- rc
        s[:, 65] = s_pockets[:, 8]
        s[:, 66] = 1.0 - s_pockets[:, 9]
        
        s[:, 67:69] = s_pockets[:, 10:12]  # lt <- rt
        s[:, 67] = s_pockets[:, 10]
        s[:, 68] = 1.0 - s_pockets[:, 11]
        
        s[:, 69:71] = s_pockets[:, 0:2]   # rb <- lb
        s[:, 69] = s_pockets[:, 0]
        s[:, 70] = 1.0 - s_pockets[:, 1]
        
        s[:, 71:73] = s_pockets[:, 2:4]   # rc <- lc
        s[:, 71] = s_pockets[:, 2]
        s[:, 72] = 1.0 - s_pockets[:, 3]
        
        s[:, 73:75] = s_pockets[:, 4:6]   # rt <- lt
        s[:, 73] = s_pockets[:, 4]
        s[:, 74] = 1.0 - s_pockets[:, 5]
        
        return s
    
    def _mirror_lr_action(self, actions):
        """左右镜像动作 (phi -> -phi)"""
        a = actions.copy()
        # phi = atan2(sin, cos)
        # -phi: sin(-phi) = -sin(phi), cos(-phi) = cos(phi)
        a[:, 1] = -a[:, 1]  # phi_sin取反
        # a[:, 2] 不变 (phi_cos)
        return a
    
    def _mirror_ud(self, states):
        """上下镜像状态 (x -> 1-x)"""
        s = states.copy()
        
        # 白球 x
        s[:, 0] = 1.0 - s[:, 0]
        
        # 15个球 x
        for i in range(15):
            idx = 3 + i * 3  # x坐标索引
            s[:, idx] = 1.0 - s[:, idx]
        
        # 袋口 - 需要交换上下袋口
        # lb(63:65) <-> lt(67:69), rb(69:71) <-> rt(73:75), lc和rc不变但x镜像
        s_pockets = s[:, 63:75].copy()
        
        # lb <-> lt
        s[:, 63] = 1.0 - s_pockets[:, 4]  # lt.x
        s[:, 64] = s_pockets[:, 5]         # lt.y
        s[:, 67] = 1.0 - s_pockets[:, 0]  # lb.x
        s[:, 68] = s_pockets[:, 1]         # lb.y
        
        # lc - x镜像
        s[:, 65] = 1.0 - s_pockets[:, 2]
        s[:, 66] = s_pockets[:, 3]
        
        # rb <-> rt
        s[:, 69] = 1.0 - s_pockets[:, 10]  # rt.x
        s[:, 70] = s_pockets[:, 11]         # rt.y
        s[:, 73] = 1.0 - s_pockets[:, 6]   # rb.x
        s[:, 74] = s_pockets[:, 7]          # rb.y
        
        # rc - x镜像
        s[:, 71] = 1.0 - s_pockets[:, 8]
        s[:, 72] = s_pockets[:, 9]
        
        return s
    
    def _mirror_ud_action(self, actions):
        """上下镜像动作 (phi -> 180-phi)"""
        a = actions.copy()
        # 180-phi: sin(180-phi) = sin(phi), cos(180-phi) = -cos(phi)
        # a[:, 1] 不变 (phi_sin)
        a[:, 2] = -a[:, 2]  # phi_cos取反
        return a
    
    def _rotate_180(self, states):
        """180度旋转 (x->1-x, y->1-y)"""
        s = states.copy()
        
        # 白球
        s[:, 0] = 1.0 - s[:, 0]
        s[:, 1] = 1.0 - s[:, 1]
        
        # 15个球
        for i in range(15):
            s[:, 3 + i * 3] = 1.0 - s[:, 3 + i * 3]      # x
            s[:, 3 + i * 3 + 1] = 1.0 - s[:, 3 + i * 3 + 1]  # y
        
        # 袋口 - 对角交换
        s_pockets = s[:, 63:75].copy()
        # lb <-> rt
        s[:, 63] = 1.0 - s_pockets[:, 10]
        s[:, 64] = 1.0 - s_pockets[:, 11]
        s[:, 73] = 1.0 - s_pockets[:, 0]
        s[:, 74] = 1.0 - s_pockets[:, 1]
        
        # lc <-> rc
        s[:, 65] = 1.0 - s_pockets[:, 8]
        s[:, 66] = 1.0 - s_pockets[:, 9]
        s[:, 71] = 1.0 - s_pockets[:, 2]
        s[:, 72] = 1.0 - s_pockets[:, 3]
        
        # lt <-> rb
        s[:, 67] = 1.0 - s_pockets[:, 6]
        s[:, 68] = 1.0 - s_pockets[:, 7]
        s[:, 69] = 1.0 - s_pockets[:, 4]
        s[:, 70] = 1.0 - s_pockets[:, 5]
        
        return s
    
    def _rotate_180_action(self, actions):
        """180度旋转动作 (phi -> phi+180)"""
        a = actions.copy()
        # phi+180: sin(phi+180) = -sin(phi), cos(phi+180) = -cos(phi)
        a[:, 1] = -a[:, 1]  # phi_sin取反
        a[:, 2] = -a[:, 2]  # phi_cos取反
        return a
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        
        # 添加状态噪声（训练时）
        if self.noise_std > 0:
            noise = torch.randn_like(state) * self.noise_std
            # 只对位置坐标添加噪声，不对pocketed标志添加噪声
            noise_mask = torch.ones_like(state)
            # 设置pocketed位置的mask为0
            noise_mask[2] = 0  # 白球pocketed
            for i in range(15):
                noise_mask[3 + i * 3 + 2] = 0  # 每个球的pocketed
            # 统计特征不加噪声
            noise_mask[75:80] = 0
            state = state + noise * noise_mask
        
        return state, action


class ImitationLoss(nn.Module):
    """模仿学习损失函数"""
    
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            'v0': 1.0,
            'phi': 2.0,
            'theta': 1.0,
            'spin': 0.5,
        }
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        v0_loss = self.mse(pred[:, 0], target[:, 0])
        
        phi_sin_loss = self.mse(pred[:, 1], target[:, 1])
        phi_cos_loss = self.mse(pred[:, 2], target[:, 2])
        phi_norm = pred[:, 1]**2 + pred[:, 2]**2
        phi_reg = self.mse(phi_norm, torch.ones_like(phi_norm))
        phi_loss = phi_sin_loss + phi_cos_loss + 0.1 * phi_reg
        
        theta_loss = self.mse(pred[:, 3], target[:, 3])
        spin_loss = self.mse(pred[:, 4:6], target[:, 4:6])
        
        total_loss = (
            self.weights['v0'] * v0_loss +
            self.weights['phi'] * phi_loss +
            self.weights['theta'] * theta_loss +
            self.weights['spin'] * spin_loss
        )
        
        return total_loss, {
            'v0': v0_loss.item(),
            'phi': phi_loss.item(),
            'theta': theta_loss.item(),
            'spin': spin_loss.item(),
        }


class Trainer:
    """训练器"""
    
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
        self.patience_counter = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        self.model.train()
        
        total_loss = 0.0
        component_losses = {'v0': 0.0, 'phi': 0.0, 'theta': 0.0, 'spin': 0.0}
        num_batches = 0
        
        for batch_idx, (states, actions) in enumerate(self.train_loader):
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(states)
                    loss, losses = self.criterion(outputs, actions)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(states)
                loss, losses = self.criterion(outputs, actions)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            for k in component_losses:
                component_losses[k] += losses[k]
            num_batches += 1
            
            if batch_idx % self.log_interval == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        for k in component_losses:
            component_losses[k] /= num_batches
        
        return avg_loss, component_losses
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        
        total_loss = 0.0
        component_losses = {'v0': 0.0, 'phi': 0.0, 'theta': 0.0, 'spin': 0.0}
        num_batches = 0
        
        for states, actions in self.val_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(states)
                    loss, losses = self.criterion(outputs, actions)
            else:
                outputs = self.model(states)
                loss, losses = self.criterion(outputs, actions)
            
            total_loss += loss.item()
            for k in component_losses:
                component_losses[k] += losses[k]
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        for k in component_losses:
            component_losses[k] /= num_batches
        
        return avg_loss, component_losses
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'model_config': {
                'state_dim': self.model.state_dim,
                'action_dim': self.model.action_dim,
                'hidden_dim': self.model.hidden_dim,
            }
        }
        
        torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_latest.pt'))
        
        if epoch % 10 == 0:
            torch.save(checkpoint, os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_best.pt'))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model_best.pt'))
    
    def train(self, num_epochs, patience=20, start_epoch=1):
        print(f"\n[INFO] Starting training for {num_epochs} epochs")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] AMP: {self.use_amp}")
        print(f"[INFO] Output: {self.output_dir}")
        
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            start_time = time.time()
            train_loss, train_components = self.train_epoch(epoch)
            train_time = time.time() - start_time
            
            val_loss, val_components = self.validate(epoch)
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  V0: {val_components['v0']:.4f} | Phi: {val_components['phi']:.4f} | "
                  f"Theta: {val_components['theta']:.4f} | Spin: {val_components['spin']:.4f}")
            print(f"LR: {current_lr:.6f} | Time: {train_time:.1f}s")
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            for k, v in val_components.items():
                self.writer.add_scalar(f'Loss_components/{k}', v, epoch)
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  ★ New best model!")
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            if self.patience_counter >= patience:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break
        
        self.writer.close()
        print(f"\n[INFO] Training completed. Best val loss: {self.best_val_loss:.4f}")
        
        return self.best_val_loss


def load_data(data_dir):
    """加载所有数据文件"""
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
        print(f"  - Loaded {sf}: {states.shape}")
    
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    print(f"[INFO] Total samples: {len(states)}")
    print(f"[INFO] State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")
    
    return states, actions


def main():
    parser = argparse.ArgumentParser(description='Train with symmetric augmentation')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data_100k',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_aug',
                        help='Output directory for checkpoints')
    
    # 增强参数
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable symmetric augmentation')
    parser.add_argument('--noise_std', type=float, default=0.01,
                        help='State noise standard deviation (0 to disable)')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='small',
                        choices=['xlarge', 'large', 'small'],
                        help='Model type')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=8,
                        help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (increased for regularization)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    
    # 硬件参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader num_workers')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载数据
    states, actions = load_data(args.data_dir)
    
    # 先划分train/val (在增强之前)
    n = len(states)
    val_size = int(n * args.val_split)
    train_size = n - val_size
    
    # 随机打乱
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_states = states[train_indices]
    train_actions = actions[train_indices]
    val_states = states[val_indices]
    val_actions = actions[val_indices]
    
    print(f"[INFO] Train samples (before aug): {len(train_states)}")
    print(f"[INFO] Val samples: {len(val_states)}")
    
    # 创建数据集 (训练集增强，验证集不增强)
    train_dataset = BilliardsAugmentedDataset(
        train_states, train_actions,
        augment=not args.no_augment,
        noise_std=args.noise_std
    )
    val_dataset = BilliardsAugmentedDataset(
        val_states, val_actions,
        augment=False,  # 验证集不增强
        noise_std=0.0
    )
    
    print(f"[INFO] Train samples (after aug): {len(train_dataset)}")
    
    # 创建DataLoader
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
    if args.model_type == 'xlarge':
        model_kwargs = {
            'state_dim': states.shape[1],
            'action_dim': actions.shape[1],
        }
    else:
        model_kwargs = {
            'state_dim': states.shape[1],
            'action_dim': actions.shape[1],
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
        }
        
        if args.model_type == 'large':
            model_kwargs['use_transformer'] = True
            model_kwargs['num_heads'] = 8
    
    model = create_model(args.model_type, **model_kwargs)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {num_params:,}")
    
    # 创建优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6,
    )
    
    # 损失函数
    criterion = ImitationLoss()
    
    # 从检查点恢复
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"[INFO] Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint.get('scheduler_state_dict') and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            print(f"[INFO] Resumed from epoch {checkpoint['epoch']}")
    
    # 创建训练器
    trainer = Trainer(
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
    
    trainer.best_val_loss = best_val_loss
    
    # 开始训练
    trainer.train(args.num_epochs, args.patience, start_epoch=start_epoch)


if __name__ == '__main__':
    main()

