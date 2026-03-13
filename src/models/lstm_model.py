import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy


class SignLanguageLSTM(pl.LightningModule):
    """
    Bidirectional LSTM with Attention for sign language recognition.
    Input  : (batch, max_frames, 258)
    Output : (batch, num_classes)
    """

    def __init__(
        self,
        input_size    = 258,
        hidden_size   = 256,    # bigger than before
        num_layers    = 3,      # deeper
        num_classes   = 20,
        dropout       = 0.3,    # less dropout (was underfitting)
        learning_rate = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # ── Input Projection ──────────────────────────────
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # ── Bidirectional LSTM ────────────────────────────
        self.lstm = nn.LSTM(
            input_size    = 256,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0,
            bidirectional = True
        )

        lstm_out_size = hidden_size * 2

        # ── Attention ─────────────────────────────────────
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # ── Classifier ────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # ── Metrics ───────────────────────────────────────
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc   = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc  = Accuracy(task='multiclass', num_classes=num_classes)

    def attention_pool(self, lstm_output):
        attn_weights = self.attention(lstm_output)
        attn_weights = F.softmax(attn_weights, dim=1)
        return (lstm_output * attn_weights).sum(dim=1)

    def forward(self, x):
        B, T, _ = x.shape
        x = x.view(B * T, -1)
        x = self.input_projection(x)
        x = x.view(B, T, -1)
        lstm_out, _ = self.lstm(x)
        pooled  = self.attention_pool(lstm_out)
        logits  = self.classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        X, y   = batch
        logits = self(X)
        # Label smoothing helps generalization on small datasets
        loss   = F.cross_entropy(logits, y, label_smoothing=0.1)
        preds  = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log('train_loss', loss,           on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc',  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y   = batch
        logits = self(X)
        loss   = F.cross_entropy(logits, y)
        preds  = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.log('val_loss', loss,         prog_bar=True)
        self.log('val_acc',  self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y   = batch
        logits = self(X)
        preds  = logits.argmax(dim=1)
        self.test_acc(preds, y)
        self.log('test_acc', self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        # Warmup for 10 epochs then cosine decay
        def lr_lambda(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, 100 - warmup_epochs)
            return max(0.01, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer"   : optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }

    def predict_sign(self, keypoints, label_map, top_k=5):
        self.eval()
        with torch.no_grad():
            x      = torch.FloatTensor(keypoints).unsqueeze(0)
            x      = x.to(next(self.parameters()).device)
            logits = self(x)
            probs  = F.softmax(logits, dim=1)[0]
            top_probs, top_indices = probs.topk(min(top_k, len(label_map)))
            return [(label_map.get(str(idx.item()), "unknown"), round(prob.item(), 4))
                    for prob, idx in zip(top_probs, top_indices)]
