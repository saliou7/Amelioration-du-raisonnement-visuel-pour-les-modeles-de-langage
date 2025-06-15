import sys
import os
import logging
import torch
from torch.utils.data import DataLoader
from PIL import Image
# Mettre à jour l'import de GradScaler
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoConfig
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.training_config import parse_args
from data.load_datasets import load_laion_220
from utils.collate_fn import collate_fn_gpt2
from utils.visualization import LearningCurveTracker
from late_fusion_model.modeling_fusion import MultimodalFusionLayer

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Chargement de la configuration
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration matérielle optimisée
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialisation du tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Chargement du modèle avec gestion mémoire avancée
    model_config = AutoConfig.from_pretrained(config.model_name_or_path)
    model = MultimodalFusionLayer.from_pretrained(
        config.model_name_or_path,
        config=model_config,
        torch_dtype=torch.float32,  # Forcer FP32 pour les paramètres
        low_cpu_mem_usage=False
    ).to(device)  
    model.train()

    # Ajout du processor pour le traitement des images
    processor = model.processor

    # Configuration DataLoader
    train_dataset = load_laion_220(config, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn_gpt2,
        shuffle=config.shuffle_data,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # Configuration de l'optimiseur uniquement pour les couches à entraîner
    trainable_params = []
    trainable_params.extend(model.vision_projector.parameters())
    trainable_params.extend(model.fusion_layer.parameters())
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # Mise à jour de l'initialisation de GradScaler
    scaler = GradScaler('cuda', enabled=config.use_fp16)
    tracker = LearningCurveTracker(save_dir=config.log_dir)

    # Variables pour la reprise d'entraînement
    starting_epoch = 0

    # Chargement d'un checkpoint si nécessaire
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint is not None and config.resume_from_checkpoint != "":
            checkpoint_path = config.resume_from_checkpoint
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Récupère le checkpoint le plus récent
            dirs = [f for f in os.scandir("outputs") if f.is_dir()]
            if not dirs:
                print("Aucun checkpoint trouvé")
            else:
                dirs.sort(key=os.path.getctime)
                path = dirs[-1].name
                checkpoint_path = os.path.join("outputs", path)

        print(f"Reprise depuis le checkpoint: {checkpoint_path}")
        # Chargement du modèle et de l'optimiseur
        checkpoint = torch.load(os.path.join(checkpoint_path, "checkpoint.pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Extraction de epoch_i ou step_i
        training_difference = os.path.splitext(path)[0]
        
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
        else:
            # Pour les checkpoints basés sur les steps
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)

    # Boucle d'entraînement optimisée
    for epoch in range(starting_epoch, config.num_train_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{config.num_train_epochs}",
            dynamic_ncols=True
        )

        for step, batch in enumerate(progress_bar):
            # Traitement des images
            images = []
            for image_path in batch['image_path']:
                image = Image.open(image_path).convert('RGB')
                images.append(image)
            
            # Création des pixel_values
            with torch.no_grad():
                pixel_values = processor(images=images, return_tensors="pt")['pixel_values']
                if config.use_fp16:
                    pixel_values = pixel_values.half()
                pixel_values = pixel_values.to(device)

            # Préparation des inputs
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
                "pixel_values": pixel_values
            }

            # Forward pass avec précision mixte
            with autocast(device_type='cuda', enabled=config.use_fp16):  # On garde autocast pour les calculs
                outputs = model(**inputs)
                loss = outputs.loss / config.gradient_accumulation_steps
                
                # Vérifier si la loss est valide
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(f"NaN ou Inf détecté dans la loss à l'étape {step}")
                    loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)
                    loss = torch.where(torch.isinf(loss), torch.full_like(loss, 1), loss)
                    continue

            # Backward pass avec gradient clipping plus agressif
            scaler.scale(loss).backward()

            # Mise à jour des poids selon l'accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                # Gradient clipping plus agressif
                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                
                # Vérifier les gradients uniquement pour les paramètres entraînables
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad:  # Ne vérifier que les paramètres entraînables
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    logger.warning(f"NaN ou Inf gradients détectés dans {name}")
                                    param.grad.zero_()

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Calcul de la loss totale
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            # Mise à jour dynamique de la barre de progression
            progress_bar.set_postfix({
                "loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                "vram": f"{torch.cuda.memory_reserved()/1e9:.2f}GB"
            })

        # Logging et sauvegarde
        avg_loss = total_loss / len(train_dataloader)
        tracker.update(epoch, avg_loss, optimizer.param_groups[0]['lr'])

        # Sauvegarde périodique
        if (epoch + 1) % config.save_every_n_epochs == 0:
            save_path = os.path.join("outputs", f"epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            
            # Sauvegarde du checkpoint pour reprise d'entraînement
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, os.path.join(save_path, "checkpoint.pt"))
            logger.info(f"Checkpoint sauvegardé → {save_path}")

        if config.plot_learning_curves and (epoch + 1) % config.plot_every_n_epochs == 0:
            tracker.plot_losses()
            tracker.plot_lr()

        logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | VRAM: {torch.cuda.memory_reserved()/1e9:.2f}GB")

    # Sauvegarde finale du modèle au format HuggingFace
    model.save_pretrained(os.path.join("outputs", "final_model"), safe_serialization=True)
    logger.info("Modèle final sauvegardé")

if __name__ == "__main__":
    main()