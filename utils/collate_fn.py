import torch

def collate_fn_gpt2(examples):
    """
    Fonction de collate pour GPT2 qui:
    1. Récupère les chemins d'images et les tokens d'entrée
    2. Aligne les séquences sur la plus longue (avant le premier token EOS)
    3. Prépare les labels et attention masks
    
    Args:
        examples: Liste de dictionnaires contenant 'image_path', 'input_ids', 'attention_mask'
    """
    # Récupération des chemins d'images
    images = [example['image_path'] for example in examples]
    
    # Conversion en tenseurs PyTorch
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)

    # Trouver où se termine la séquence réelle (premier token EOS = 50256)
    eos_positions = (input_ids == 50256).long().argmax(dim=1)
    max_length = eos_positions.max() + 1  # +1 pour inclure le token EOS
    
    # Tronquer les séquences à la longueur max utile
    input_ids = input_ids[:, :max_length]
    attention_mask = attention_mask[:, :max_length]

    # Si des crops sont présents, les inclure
    if 'crop' in examples[0]:
        crops = [example["crop"] for example in examples]
        return {
            "image_path": images,
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # Pour l'entraînement auto-régressif
            "attention_mask": attention_mask,
            "crop": crops,
        }

    return {
        "image_path": images,
        "input_ids": input_ids,
        "labels": input_ids.clone(),  # Pour l'entraînement auto-régressif
        "attention_mask": attention_mask,
    }
