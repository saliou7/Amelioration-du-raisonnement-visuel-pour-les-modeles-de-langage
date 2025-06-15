import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from nltk import sent_tokenize
from torch.utils.data import DataLoader
from collections import Counter
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from transformers import AutoTokenizer, AutoConfig
from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference import load_checkpoint, compute_scores
from config.testing_config import parse_args

def load_candidates(subset):
    if subset == 'color':
        candidates = ['brown', 'black', 'white', 'yellow', 'green', 'gray', 'red', 'orange', 'blue', 'silver','pink']
    elif subset == 'shape':
        candidates = ['round', 'rectangle', 'triangle', 'square', 'oval', 'curved', 'cylinder', 'straight',
                      'cone', 'curly', 'heart', 'star']
    elif subset == 'material':
        candidates = ['metal', 'wood', 'plastic', 'cotton', 'glass', 'fabric', 'stone', 'rubber', 'ceramic',
                      'cloth', 'leather', 'flour', 'paper', 'clay', 'wax', 'concrete']
    elif subset == 'component':
        candidates = ['yes', 'no']
    elif subset == 'others_yes':
        candidates = ['yes', 'no']
    elif subset == 'others_number':
        candidates = ['2', '4', '6', '1', '8', '3', '5']
    elif subset == 'others':
        candidates = ['long', 'small', 'short', 'large', 'forest', 'water', 'ocean', 'big', 'tree', 'ground', 'tall',
                      'wild', 'outside', 'thin', 'head', 'thick', 'circle', 'brown', 'soft', 'land', 'neck', 'rough',
                      'chest', 'smooth', 'fur', 'hard', 'top', 'plants', 'black', 'metal', 'books', 'vertical', 'lake',
                      'grass', 'road', 'sky', 'front', 'kitchen', 'feathers', 'stripes', 'baby', 'hair', 'feet',
                      'mouth', 'female', 'table']
    else:
        raise ValueError(f"Subset {subset} does not exist!")
    return candidates

def format_question(question, prompt_idx=0):
    prompts = [
        f"{question}",
        f"{question} Answer:",
        f"{question} The answer is",
        f"Question: {question} Answer:",
        f"Question: {question} The answer is"
    ]
    return prompts[prompt_idx]

def preprocess_candidates(candidates, tokenizer):
    processed = []
    for cand in candidates:
        # Try different variations
        variants = [
            tokenizer.encode(' ' + cand, add_special_tokens=False),  # Space-prefixed
            tokenizer.encode(cand, add_special_tokens=False),        # Original
            tokenizer.encode(cand.capitalize(), add_special_tokens=False)  # Capitalized
        ]
        
        # Find the shortest valid tokenization
        valid_variants = [v for v in variants if len(v) > 0]
        if not valid_variants:
            raise ValueError(f"Could not tokenize candidate: {cand}")
            
        shortest = min(valid_variants, key=len)
        processed.append((cand, shortest))
        
        # Debug print
        print(f"Candidate: {cand:<10} | Tokens: {shortest} | As text: {[tokenizer.decode(t) for t in shortest]}")
    
    return processed



def main():
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_checkpoint(config.path)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    t2i_model = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    

    t2i_model.safety_checker = None

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data/imageNetVC/")

    # Load dataset
    df = pd.read_csv(os.path.join(data_dir, f"{config.subset}.csv"), header=0)
    validation_data = [
        {
            'input_text': format_question(row['question'], config.prompt_idx),
            'label': str(row['answer']).lower()
        }
        for _, row in df.iterrows()
    ][:config.num_test_samples]

    validation_loader = DataLoader(validation_data, batch_size=1)

    # Load candidates and token IDs
    candidates = load_candidates(config.subset)
    processed_candidates = preprocess_candidates(candidates, tokenizer)
    
    # Evaluation loop
    predictions = []
    true_labels = []
    avg_clip_scores = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Evaluating..."):
            input_text = batch['input_text'][0]
            label = batch['label'][0]

            print()
            print("Input text:", input_text)
            print()

            pixel_values = None
            if config.generate_images:
                # Extract question part for image prompts
                question_part = input_text.split("Answer:")[0].replace("Question: ", "").strip()
                sentences = sent_tokenize(question_part)
                prompts = [question_part]
                if config.k > 1:
                    prompts += [sentences[-1] if sentences else question_part for _ in range(config.k - 1)]
                
                # Generate images
                images = []
                valid_prompts = [p for p in prompts if p.strip()]
                if not valid_prompts:
                    pixel_values = None
                else:
                    images = t2i_model(valid_prompts, num_inference_steps=50, guidance_scale=7.5, generator=torch.manual_seed(42)).images
                    pixel_values = clip_processor(images=images, return_tensors="pt")['pixel_values'].to(device).to(model.dtype)
                    

                # Display images
                os.makedirs(f"output/images/imageNetVC/{config.subset}", exist_ok=True)
                if config.show_images:
                    for i, image in enumerate(images):
                        image.show()
                        if config.save_images:
                            image.save(f"output/images/imageNetVC/{config.subset}/img_{i}.png")
                        
            
            
            # Prepare inputs
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            if pixel_values is not None:
                inputs['pixel_values'] = pixel_values
                inputs['scores'] = compute_scores(clip_model, inputs['pixel_values'], inputs['input_ids'])
                print("Scores:", inputs['scores'])
                avg_clip_scores.append(inputs['scores'].mean().item())
            
            # Model inference
            output = model(**inputs)
            logits = output.logits

            # Calculate scores for each candidate sequence
            candidate_scores = []
            for cand_text, token_sequence in processed_candidates:
                total_score = 0.0
                current_input = inputs.input_ids.clone()
                
                try:
                    for token_id in token_sequence:
                        # Get logits for the next position
                        next_logits = model(current_input).logits[:, -1, :]
                        
                        # Get score for this token
                        token_score = next_logits[0, token_id].item()
                        total_score += token_score
                        
                        # Append token to input for next step
                        current_input = torch.cat([
                            current_input, 
                            torch.tensor([[token_id]]).to(device)
                        ], dim=-1)
                        
                    candidate_scores.append((cand_text, total_score))
                
                except Exception as e:
                    print(f"Error processing {cand_text}: {str(e)}")
                    candidate_scores.append((cand_text, -float('inf')))

            # Get best candidate
            if candidate_scores:
                predicted_candidate = max(candidate_scores, key=lambda x: x[1])[0]
            else:
                predicted_candidate = "unknown"

            predictions.append(predicted_candidate)
            true_labels.append(label)
        
            print()
            print("Prediction:", predicted_candidate,"\n")
            print("True label:", label)
            print()
    
    # Compute accuracy
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"Accuracy: {accuracy:.3f}")
    if config.generate_images:
        avg_clip_scores = np.mean(avg_clip_scores)
        print(f"Avg Clip Score: {0:.3f}")

    # Save results
    results_path = f"output/results/imageNetVC/{config.subset}_evaluation.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'a') as f:
        if not os.path.exists(results_path):
            f.write("Data set,Subset,K,Batch Size,Num Test Samples,Avg Clip Score,Accuracy\n")
        avg_clip_score_str = f"{avg_clip_scores:.3f}" if config.generate_images else "nan"
        f.write(f"ImageNetVC,{config.subset},{config.k if config.generate_images else 0},{config.batch_size},{config.num_test_samples},{avg_clip_score_str},{accuracy:.3f}\n")

if __name__ == "__main__":
    main()