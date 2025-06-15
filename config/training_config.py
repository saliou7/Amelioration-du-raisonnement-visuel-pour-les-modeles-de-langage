from dataclasses import dataclass
from typing import Optional
import argparse
@dataclass
class TrainingConfig:
    # Configuration du modèle
    model_name_or_path: str = "gpt2"
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    
    # Configuration de l'entraînement
    batch_size: int = 2                  # Batch size plus petit pour commencer
    gradient_accumulation_steps: int = 32  # Plus d'accumulation
    learning_rate: float = 5e-5        # Learning rate encore plus petit
    max_grad_norm: float = 1.0           # Clipping encore plus agressif
    warmup_steps: int = 100              # Ajouter du warmup
    weight_decay: float = 0.01           # Régularisation plus forte
    num_train_epochs: int = 10
    max_train_steps: Optional[int] = None
    
    # Configuration du dataset
    max_seq_length: int = 1024
    max_train_samples: Optional[int] = None
    shuffle_data: bool = True
    short: bool = True
    
    # Configuration de la précision
    run_bf16: bool = False
    use_fp16: bool = True 

    # Configuration du logging et visualisation
    plot_every_n_epochs: int = 5
    save_every_n_epochs: int = 1
    log_dir: str = "outputs/logs"
    plot_learning_curves: bool = False
    num_workers: int = 4   

    overwrite_cache: bool = False
    resume_from_checkpoint: Optional[str] = None # Pour reprendre depuis un checkpoint
default_config = TrainingConfig()

def parse_args():
    parser = argparse.ArgumentParser(description="Training arguments")
    config = default_config  # On commence avec la config par défaut
    
    # Ajout des arguments basés sur la config par défaut
    for key, value in vars(config).items():
        arg_type = type(value) if value is not None else str
        parser.add_argument(
            f"--{key}",
            type=arg_type,
            default=value,
            help=f"Default: {value}"  # Affiche la valeur par défaut dans l'aide
        )

    args = parser.parse_args()
    # Création d'une nouvelle config avec les valeurs de la ligne de commande
    training_config = TrainingConfig(**vars(args))
    return training_config
