"""DiaTrust-NX Advanced Artificial Neural Network (ANN) Architectures
Implements 8 distinct PyTorch deep learning architectures for diabetes risk estimation:
1. ANN Baseline
2. Deep ANN
3. Wide ANN
4. Regularized ANN
5. Batch Normalization ANN
6. Dropout ANN
7. Residual MLP
8. Optimized ANN with Learning-Rate Scheduling & Class-Imbalance Handling
"""

import copy
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin


# ---------------------------------------------------------
# PyTorch Neural Network Architectures
# ---------------------------------------------------------

class BaselineANN(nn.Module):
    """Architecture 1: ANN Baseline (2-layer MLP)."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DeepANN(nn.Module):
    """Architecture 2: Deep ANN (5 hidden layers)."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WideANN(nn.Module):
    """Architecture 3: Wide ANN (256 -> 128 hidden units)."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RegularizedANN(nn.Module):
    """Architecture 4: Regularized ANN (L2 Weight Decay + Dropout 0.3)."""
    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(48, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BatchNormANN(nn.Module):
    """Architecture 5: Batch Normalization ANN."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.out(x).squeeze(-1)


class DropoutANN(nn.Module):
    """Architecture 6: High Dropout ANN (Dropout 0.45)."""
    def __init__(self, input_dim: int, dropout_rate: float = 0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResMLPBlock(nn.Module):
    """Residual MLP Block with projection, batchnorm and skip-connection."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = F.relu(out + residual)
        return out


class ResidualMLP(nn.Module):
    """Architecture 7: Residual MLP with skip connections."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_blocks: int = 2, dropout: float = 0.2):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([ResMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


class OptimizedANN(nn.Module):
    """Architecture 8: Optimized ANN with dynamic residual routing and normalization."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, 96)
        self.bn1 = nn.BatchNorm1d(96)
        
        self.dense1 = nn.Linear(96, 96)
        self.bn2 = nn.BatchNorm1d(96)
        self.drop1 = nn.Dropout(0.25)
        
        self.dense2 = nn.Linear(96, 48)
        self.bn3 = nn.BatchNorm1d(48)
        self.drop2 = nn.Dropout(0.20)
        
        self.out = nn.Linear(48, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = F.relu(self.bn1(self.in_proj(x)))
        x1 = F.relu(self.bn2(self.dense1(x0)))
        x1 = self.drop1(x1) + x0  # Residual addition
        x2 = F.relu(self.bn3(self.dense2(x1)))
        x2 = self.drop2(x2)
        return self.out(x2).squeeze(-1)


# ---------------------------------------------------------
# Focal Loss for Class Imbalance
# ---------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    """Focal Loss for addressing severe class imbalance (e.g. CDC dataset)."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_weight * bce_loss
        else:
            loss = focal_weight * bce_loss
            
        return loss.mean()


# ---------------------------------------------------------
# Scikit-Learn Compatible PyTorch ANN Wrapper
# ---------------------------------------------------------

class PyTorchANNClassifier(BaseEstimator, ClassifierMixin):
    """Unified Scikit-Learn compatible estimator wrapping all 8 PyTorch ANN architectures.
    
    Includes:
    - Early stopping on validation loss
    - Learning rate scheduling (CosineAnnealingLR or ReduceLROnPlateau)
    - Full training and validation loss / accuracy history tracking
    - Model checkpointing
    - Automatic class weight adjustment for imbalance
    """
    
    ARCH_MAP = {
        "baseline": BaselineANN,
        "deep": DeepANN,
        "wide": WideANN,
        "regularized": RegularizedANN,
        "batch_norm": BatchNormANN,
        "dropout": DropoutANN,
        "residual_mlp": ResidualMLP,
        "optimized": OptimizedANN
    }

    def __init__(
        self,
        arch: str = "optimized",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 60,
        patience: int = 10,
        use_focal_loss: bool = False,
        use_class_weight: bool = True,
        use_lr_scheduler: bool = True,
        random_state: int = 42
    ):
        self.arch = arch
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.use_focal_loss = use_focal_loss
        self.use_class_weight = use_class_weight
        self.use_lr_scheduler = use_lr_scheduler
        self.random_state = random_state

        self.model_ = None
        self.history_: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        self.best_epoch_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        input_dim = X.shape[1]
        arch_class = self.ARCH_MAP.get(self.arch.lower())
        if arch_class is None:
            raise ValueError(f"Unknown architecture '{self.arch}'. Available: {list(self.ARCH_MAP.keys())}")
            
        self.model_ = arch_class(input_dim)
        
        # Determine positive class weight
        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        pos_weight_val = (neg_count / max(1, pos_count)) if self.use_class_weight else 1.0
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32)

        if self.use_focal_loss:
            criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
        else:
            scheduler = None

        # Prepare Tensors
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        N = len(X_t)
        effective_batch_size = min(self.batch_size, N)

        has_val = eval_set is not None
        if has_val:
            X_val_t = torch.tensor(eval_set[0], dtype=torch.float32)
            y_val_t = torch.tensor(eval_set[1], dtype=torch.float32)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        self.history_ = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            running_loss = 0.0
            correct_train = 0

            perm = torch.randperm(N)
            for i in range(0, N, effective_batch_size):
                batch_idx = perm[i:i + effective_batch_size]
                batch_X = X_t[batch_idx]
                batch_y = y_t[batch_idx]

                optimizer.zero_grad()
                logits = self.model_(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(batch_y)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_train += (preds == batch_y).sum().item()

            train_loss = running_loss / N
            train_acc = correct_train / N
            self.history_["train_loss"].append(train_loss)
            self.history_["train_acc"].append(train_acc)

            # Validation
            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    val_logits = self.model_(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                    val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
                    val_acc = (val_preds == y_val_t).sum().item() / len(y_val_t)

                self.history_["val_loss"].append(val_loss)
                self.history_["val_acc"].append(val_acc)

                if scheduler is not None:
                    scheduler.step(val_loss)

                # Checkpointing and Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model_.state_dict())
                    self.best_epoch_ = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break
            else:
                self.best_epoch_ = epoch

        # Load best checkpoint if validation set was used
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model_(X_t)
            probs_pos = torch.sigmoid(logits).cpu().numpy().flatten()
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

