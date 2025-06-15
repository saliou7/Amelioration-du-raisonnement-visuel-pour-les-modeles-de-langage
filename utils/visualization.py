import matplotlib
matplotlib.use('Agg')  # Utiliser le backend 'Agg' qui ne requiert pas de GUI
import matplotlib.pyplot as plt
from pathlib import Path
import json

class LearningCurveTracker:
    def __init__(self, save_dir: str = "outputs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
        
    def update(self, epoch: int, train_loss: float, val_loss: float = None, lr: float = None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)
            
    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.epochs, self.val_losses, label='Validation Loss')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / 'learning_curves.png')
        plt.close()
        
    def plot_lr(self):
        if self.learning_rates:
            plt.figure(figsize=(10, 5))
            plt.plot(self.epochs, self.learning_rates)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig(self.save_dir / 'lr_schedule.png')
            plt.close()
            
    def save_stats(self):
        stats = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'epochs': self.epochs
        }
        with open(self.save_dir / 'training_stats.json', 'w') as f:
            json.dump(stats, f)
            
    def load_stats(self):
        try:
            with open(self.save_dir / 'training_stats.json', 'r') as f:
                stats = json.load(f)
            self.train_losses = stats['train_losses']
            self.val_losses = stats['val_losses']
            self.learning_rates = stats['learning_rates']
            self.epochs = stats['epochs']
        except FileNotFoundError:
            print("No previous training stats found.")
