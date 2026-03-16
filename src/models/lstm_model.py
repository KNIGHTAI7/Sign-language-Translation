import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SignLanguageLSTM(pl.LightningModule):
    """
    Transformer for Sign Language Recognition with signer-invariant features.

    Input  : (batch, max_frames, 243)
    Output : (batch, num_classes)
    """

    def __init__(
        self,
        input_size    = 243,
        hidden_size   = 256,
        num_layers    = 4,
        num_classes   = 20,
        dropout       = 0.2,
        learning_rate = 3e-4,
        nhead         = 4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        d_model = hidden_size

        # input_size * 2 because we add velocity
        self.input_proj = nn.Sequential(
            nn.Linear(input_size * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.cls_token    = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            norm       = nn.LayerNorm(d_model)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.train_acc    = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.val_acc      = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.val_acc_top3 = Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.test_acc     = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.test_acc_top3= Accuracy(task='multiclass', num_classes=num_classes, top_k=3)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'cls_token' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_velocity(self, x):
        v = torch.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        return torch.cat([x, v], dim=-1)

    def forward(self, x):
        B = x.size(0)
        x = self.add_velocity(x)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.transformer(x)
        return self.classifier(x[:, 0])

    def mixup(self, x, y, alpha=0.3):
        lam  = np.random.beta(alpha, alpha) if alpha > 0 else 1
        idx  = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    def training_step(self, batch, batch_idx):
        X, y = batch
        if torch.rand(1).item() < 0.5:
            X, ya, yb, lam = self.mixup(X, y)
            logits = self(X)
            loss   = lam * F.cross_entropy(logits, ya, label_smoothing=0.1) + \
                     (1 - lam) * F.cross_entropy(logits, yb, label_smoothing=0.1)
        else:
            logits = self(X)
            loss   = F.cross_entropy(logits, y, label_smoothing=0.1)
        self.train_acc(logits.argmax(1), y)
        self.log('train_loss', loss,           on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc',  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y   = batch
        logits = self(X)
        loss   = F.cross_entropy(logits, y)
        self.val_acc(logits.argmax(1), y)
        self.val_acc_top3(logits, y)
        self.log('val_loss',     loss,              prog_bar=True)
        self.log('val_acc',      self.val_acc,      prog_bar=True)
        self.log('val_acc_top3', self.val_acc_top3, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y   = batch
        logits = self(X)
        self.test_acc(logits.argmax(1), y)
        self.test_acc_top3(logits, y)
        self.log('test_acc',      self.test_acc)
        self.log('test_acc_top3', self.test_acc_top3)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate,
            weight_decay=1e-2, betas=(0.9, 0.98)
        )
        def lr_lambda(epoch):
            warmup = 15
            if epoch < warmup:
                return (epoch + 1) / warmup
            p = (epoch - warmup) / max(1, 200 - warmup)
            return max(0.05, 0.5 * (1 + math.cos(math.pi * p)))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def predict_sign(self, keypoints, label_map, top_k=5):
        self.eval()
        with torch.no_grad():
            x      = torch.FloatTensor(keypoints).unsqueeze(0).to(next(self.parameters()).device)
            logits = self(x)
            probs  = F.softmax(logits, dim=1)[0]
            top_p, top_i = probs.topk(min(top_k, len(label_map)))
            return [(label_map.get(str(i.item()), "unknown"), round(p.item(), 4))
                    for p, i in zip(top_p, top_i)]
