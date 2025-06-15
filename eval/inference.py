import torch
from PIL import Image
from transformers import AutoTokenizer, AutoConfig
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from late_fusion_model.modeling_fusion import MultimodalFusionLayer
import torch.nn.functional as F

def compute_scores(clip_model, clip_processor, images, prompts):
    # Prepare inputs for the CLIP model using the processor
    processed_inputs = clip_processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(clip_model.device)

    # Extract image and text embeddings without calculating gradients
    with torch.no_grad():
        model_outputs = clip_model(**processed_inputs)
        image_embeddings = model_outputs.image_embeds
        text_embeddings = model_outputs.text_embeds

    # Calculate cosine similarity using PyTorch's functional API
    cosine_similarities = F.cosine_similarity(image_embeddings, text_embeddings, dim=-1)

    # Replace negative similarities with zero
    positive_similarities = F.relu(cosine_similarities)

    return positive_similarities

def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Charge le modèle à partir d'un checkpoint"""
    # Initialisation du modèle et du tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configuration et modèle
    model_config = AutoConfig.from_pretrained("gpt2")
    model = MultimodalFusionLayer.from_pretrained(
        "gpt2",
        config=model_config,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Chargement du checkpoint avec vérification
    checkpoint_file = os.path.join(checkpoint_path, "model.pt")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_file}")
    
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    if 'model_state_dict' not in checkpoint:
        raise KeyError("Le checkpoint ne contient pas model_state_dict")
    
    # Vérifier que tous les paramètres sont chargés
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'])
    if len(missing_keys) > 0:
        print(f"Attention: Clés manquantes: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Attention: Clés inattendues: {unexpected_keys}")
    
    model.eval()
    return model, tokenizer
