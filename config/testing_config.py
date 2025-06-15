from dataclasses import dataclass
from typing import Optional
import argparse

@dataclass

class TestingConfig:
    # Configuration du modèle
    model_name: str = "MultimodalFusionLayer_2"
    model_name_or_path: str = "gpt2" 
    path: str = "/home/saliou/amal/sLMIG/outputs/" 

    # Configuration des tests
    batch_size: int = 1     
    max_new_tokens: int = 15 # squad 
    predictions_dir: str = "outputs/predictions"  
    
    # nb of images to generate
    k: int = 1
    generate_images: bool = False


    # Configuration du dataset
    subset: str = "others" # imagenetvc   
    prompt_idx: int = 3   # imagenetvc  

    
    # nb of test samples
    num_test_samples: int = 5

    # show images
    show_images: bool = False
    save_images: bool = False
    
     

# Config par défaut pour les tests
default_config = TestingConfig()

def parse_args():
    parser = argparse.ArgumentParser(description="Testing arguments")
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
    testing_config = TestingConfig(**vars(args))
    return testing_config
