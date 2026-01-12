#!/usr/bin/env python3
"""
train.py - 模仿学习训练脚本

支持:
- RTX 6000 Ada GPU训练
- 混合精度训练 (AMP)
- 学习率调度
- 早停
- 模型检查点
- TensorBoard日志

使用方法:
    python train.py --data_dir ./data --output_dir ./checkpoints
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

from model import BilliardsPolicyNetwork, BilliardsPolicyNetworkSmall, create_model


class BilliardsDataset(Dataset):
    """台球数据集"""
    
    def __init__(self, states, actions):
        """
        参数:
            states: np.ndarray [N, state_dim]
            actions: np.ndarray [N, action_dim]
        """
        self.states = torch.from_numpy(states).float()
        self.actions = torch.from_numpy(actions).float()
        
        assert len(self.states) == len(self.actions)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def load_data(data_dir):
    """
    加载所有数据文件
    
    返回: states, actions (numpy arrays)
    """
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


class ImitationLoss(nn.Module):
    """
    模仿学习损失函数
    
    针对不同输出分量使用不同的损失:
    - V0: MSE
    - phi (sin/cos): MSE + 角度一致性正则
    - theta: MSE
    - a, b: MSE
    """
    
    def __init__(self, weights=None):
        super().__init__()
        
        # 各分量权重
        self.weights = weights or {
            'v0': 1.0,
            'phi': 2.0,      # 角度更重要
            'theta': 1.0,
            'spin': 0.5,     # 旋杆相对不那么关键
        }
        
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        pred, target: [B, 6] - [V0, phi_sin, phi_cos, theta, a, b]
        """
        # 速度损失
        v0_loss = self.mse(pred[:, 0], target[:, 0])
        
        # 角度损失 (phi)
        phi_sin_loss = self.mse(pred[:, 1], target[:, 1])
        phi_cos_loss = self.mse(pred[:, 2], target[:, 2])
        
        # 确保sin^2 + cos^2 ≈ 1 的正则项
        phi_norm = pred[:, 1]**2 + pred[:, 2]**2
        phi_reg = self.mse(phi_norm, torch.ones_like(phi_norm))
        
        phi_loss = phi_sin_loss + phi_cos_loss + 0.1 * phi_reg
        
        # theta损失
        theta_loss = self.mse(pred[:, 3], target[:, 3])
        
        # 旋杆损失
        spin_loss = self.mse(pred[:, 4:6], target[:, 4:6])
        
        # 加权总损失
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
        
        # 混合精度
        self.scaler = GradScaler() if use_amp else None
        
        # TensorBoard
        self.writer = SummaryWriter(os.path.join(output_dir, 'logs'))
        
        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
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
        """验证"""
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
        """保存检查点"""
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
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_latest.pt'))
        
        # 定期保存
        if epoch % 10 == 0:
            torch.save(checkpoint, os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_best.pt'))
            # 同时保存纯模型权重（方便部署）
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model_best.pt'))
    
    def train(self, num_epochs, patience=20, start_epoch=1):
        """
        完整训练流程
        
        参数:
            num_epochs: 最大训练轮数
            patience: 早停耐心值
            start_epoch: 起始epoch (用于恢复训练)
        """
        print(f"\n[INFO] Starting training for {num_epochs} epochs")
        if start_epoch > 1:
            print(f"[INFO] Resuming from epoch {start_epoch}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] AMP: {self.use_amp}")
        print(f"[INFO] Output: {self.output_dir}")
        
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # 训练
            start_time = time.time()
            train_loss, train_components = self.train_epoch(epoch)
            train_time = time.time() - start_time
            
            # 验证
            val_loss, val_components = self.validate(epoch)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印结果
            print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  V0: {val_components['v0']:.4f} | Phi: {val_components['phi']:.4f} | "
                  f"Theta: {val_components['theta']:.4f} | Spin: {val_components['spin']:.4f}")
            print(f"LR: {current_lr:.6f} | Time: {train_time:.1f}s")
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            for k, v in val_components.items():
                self.writer.add_scalar(f'Loss_components/{k}', v, epoch)
            
            # 检查是否是最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  ★ New best model!")
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # 早停检查
            if self.patience_counter >= patience:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break
        
        self.writer.close()
        print(f"\n[INFO] Training completed. Best val loss: {self.best_val_loss:.4f}")
        
        return self.best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train imitation learning model')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='large',
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
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
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
                        help='Path to checkpoint to resume from (e.g., ./checkpoints/checkpoint_latest.pt)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 加载数据
    states, actions = load_data(args.data_dir)
    
    # 创建数据集
    dataset = BilliardsDataset(states, actions)
    
    # 划分训练/验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples: {len(val_dataset)}")
    
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
        # xlarge 使用预设的超大参数配置
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
            print(f"[INFO] Previous val loss: {checkpoint['val_loss']:.4f}")
        else:
            print(f"[WARNING] Checkpoint not found: {args.resume}")
            print("[INFO] Starting from scratch")
    
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
    
    # 设置恢复状态
    trainer.best_val_loss = best_val_loss
    
    # 开始训练
    trainer.train(args.num_epochs, args.patience, start_epoch=start_epoch)


if __name__ == '__main__':
    main()



