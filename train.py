import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from data.dataset import get_dataloaders
from models.lstm_model import SignLanguageLSTM

CONFIG = {
    'processed_dir' : 'data/processed',
    'batch_size'    : 16,
    'max_frames'    : 30,
    'max_epochs'    : 200,
    'learning_rate' : 1e-3,
    'hidden_size'   : 256,
    'num_layers'    : 3,
    'dropout'       : 0.3,
    'num_workers'   : 0,
    'seed'          : 42,
}

def train():
    pl.seed_everything(CONFIG['seed'])

    print("=" * 50)
    print("Loading data...")
    print("=" * 50)

    train_loader, val_loader, test_loader, label_map = get_dataloaders(
        processed_dir = CONFIG['processed_dir'],
        batch_size    = CONFIG['batch_size'],
        max_frames    = CONFIG['max_frames'],
        num_workers   = CONFIG['num_workers']
    )

    num_classes = len(label_map)
    sample_X, _ = next(iter(train_loader))
    input_size  = sample_X.shape[-1]

    print(f"\nClasses    : {num_classes}")
    print(f"Input size : {input_size}")
    print(f"Train      : {len(train_loader.dataset)}")
    print(f"Val        : {len(val_loader.dataset)}")
    print(f"Test       : {len(test_loader.dataset)}")

    print("\n" + "=" * 50)
    print("Building model...")
    print("=" * 50)

    model = SignLanguageLSTM(
        input_size    = input_size,
        hidden_size   = CONFIG['hidden_size'],
        num_layers    = CONFIG['num_layers'],
        num_classes   = num_classes,
        dropout       = CONFIG['dropout'],
        learning_rate = CONFIG['learning_rate']
    )

    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

    os.makedirs('models/checkpoints', exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath='models/checkpoints',
            filename='best-{epoch:02d}-{val_acc:.3f}',
            monitor='val_acc', mode='max',
            save_top_k=3, verbose=True
        ),
        EarlyStopping(
            monitor='val_acc', patience=30,
            mode='max', verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    trainer = pl.Trainer(
        max_epochs          = CONFIG['max_epochs'],
        callbacks           = callbacks,
        logger              = CSVLogger('logs', name='sign_lstm'),
        accelerator         = 'gpu' if torch.cuda.is_available() else 'cpu',
        devices             = 1,
        precision           = '16-mixed',
        log_every_n_steps   = 5,
        enable_progress_bar = True
    )

    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 50)
    print("Running test evaluation...")
    print("=" * 50)
    trainer.test(model, test_loader)

    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict' : model.state_dict(),
        'label_map'        : label_map,
        'config'           : CONFIG,
        'input_size'       : input_size,
        'num_classes'      : num_classes
    }, 'models/sign_language_lstm.pth')

    print("\n✅ Training complete!")
    print(f"   Best val_acc  → check models/checkpoints/")
    print(f"   Final model   → models/sign_language_lstm.pth")

if __name__ == "__main__":
    train()
