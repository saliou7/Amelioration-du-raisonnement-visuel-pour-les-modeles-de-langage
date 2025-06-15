import os
import sys
import torch
import numpy as np
from tqdm.auto import tqdm
from nltk import sent_tokenize
from torch.utils.data import DataLoader
from collections import Counter
from transformers import CLIPModel, CLIPProcessor
import random

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from diffusers import AutoPipelineForText2Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from late_fusion_model.modeling_fusion import MultimodalFusionLayer
from config.testing_config import parse_args
from inference import load_checkpoint, compute_scores



def main():
    config = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer=load_checkpoint(config.path, device=device)
    if config.generate_images : 
        # Load the text-to-image model
        t2i_model = AutoPipelineForText2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        t2i_model.safety_checker = None  # Disable the safety checker

        # Load the CLIP model for alignment scoring
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_tokenizer = clip_processor.tokenizer # Get the tokenizer

    # Load the BoolQ dataset
    dataset = load_dataset('google/boolq')
    validation_set = dataset['validation']
    validation_set = validation_set[:config.num_test_samples]  # Use only the first n samples

    # Prepare validation data
    validation_data = [
        {
            'input_text': f"Context: {validation_set['passage'][i].capitalize()}. Question: {validation_set['question'][i].capitalize()}? Answer:",
            'label': 'yes' if validation_set['answer'][i] else 'no'
        }
        for i in range(len(validation_set['question']))
    ]

    validation_loader = DataLoader(validation_data, batch_size=1)

    # Define special tokens
    yes_token = 'yes'
    no_token = 'no'
    yes_token_id = tokenizer.encode(yes_token, add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(no_token, add_special_tokens=False)[0]

    # Debug: Print token IDs for 'yes' and 'no'
    # print(f"Token ID for 'yes': {yes_token_id}")
    # print(f"Token ID for 'no': {no_token_id}")

    # Evaluate the model
    predictions , true_labels= [],[]
    avg_clip_scores = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Evaluating..."):
            input_text = batch['input_text'][0]
            label = batch['label'][0]
            inputs = {}

            # print(f"Input text: {input_text}")

            pixel_values = None
            if config.generate_images:

                # Generate k prompts for the input text
                prompts = []
                if config.k > 0:
                    prompts = [' '.join(input_text.split()[-100:])]  # Use the last m words as the prompt
                if config.k > 1:
                    prompts += [sent_tokenize(input_text)[-1] for _ in range(config.k - 1)]
                
                # Debug: Print prompts to verify their contents
                # print(f"Prompts: {prompts}")

                # Truncate prompts using CLIP's tokenizer to 77 tokens newww
                truncated_prompts = []
                for prompt in prompts:
                    # Tokenize with CLIP's tokenizer and truncate
                    inputs = clip_tokenizer(prompt,max_length=77,truncation=True,return_tensors="pt",padding="max_length")

                    # Decode back to text (without special tokens)
                    truncated_prompt = clip_tokenizer.decode(inputs['input_ids'][0],skip_special_tokens=True)
                    truncated_prompts.append(truncated_prompt)
                prompts = truncated_prompts  # Use truncated prompts 
                
                # Skip if no valid prompts are generated
                if not prompts or all(p.strip() == '' for p in prompts) :
                    print("Warning: No valid prompts generated. Skipping image generation.")
                    pixel_values = None
                else:
                    # Generate images for all prompts
                    images = t2i_model(prompts, num_inference_steps=50, guidance_scale=7.5, generator=torch.manual_seed(42)).images

                    os.makedirs("output/images/boolq/", exist_ok=True)
                    # Display images
                    
                    for i, image in enumerate(images):
                        if config.show_images:
                            image.show()
                        if config.save_images:
                            image.save(f"output/images/boolq/img_{i}.png")
                    

                    # Process images into tensor format for the model
                    pixel_values = clip_processor(images=images, return_tensors="pt").data['pixel_values']
                    inputs['pixel_values'] = pixel_values.to(device).to(model.dtype)
                    # print("pixel value")
                    # Calculate CLIP alignment scores
                    clip_scores = compute_scores(clip_model, clip_processor, images, prompts)
                    inputs['scores'] = clip_scores
                    avg_clip_scores.append(clip_scores.mean().item())
                    print(f"Average CLIP score: {clip_scores.mean().item()}")

            # Tokenize input text
            tokenized_inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs.update(tokenized_inputs)

            # Model inference
            output = model(**inputs)

            # Extract logits and apply masking
            logits = output.logits[:, -1, :]  # Get logits of the last token

            # Mask all tokens except 'yes' and 'no'
            mask = torch.ones_like(logits) * float('-inf')
            mask[:, yes_token_id] = logits[:, yes_token_id]
            mask[:, no_token_id] = logits[:, no_token_id]
            logits = mask

            # Compute probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # Aggregate predictions for all k prompts
            predicted_token_ids = probabilities.argmax(dim=-1).tolist()
            predictions_for_instance = [tokenizer.decode([token_id]).strip().lower() for token_id in predicted_token_ids]

            # Use majority voting to determine the final prediction
            final_prediction = Counter(predictions_for_instance).most_common(1)[0][0]

            # Debug: Print predictions and final decision
            print(f"Predictions for instance: {predictions_for_instance}")
            print(f"Final prediction: {final_prediction}, True Label: {label}")

            # Append final prediction and true label
            predictions.append(final_prediction)
            true_labels.append(label)

    # accuracy
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"Accuracy: {accuracy:.3f}")
    if config.generate_images:
        avg_clip_score = np.mean(0)
        print(f"Average CLIP score: {avg_clip_score:.3f}")

     # Save results
    results_path = f"output/results/boolq_evaluation.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Check if the file already exists
    file_exists = os.path.exists(results_path)

    with open(results_path, 'a') as file:
        # Write the header only if the file does not exist
        if not file_exists:
            file.write("Dataset,K,Batch Size,Num Test Samples,Avg Clip Score,Accuracy\n")
        # Write the results
        avg_clip_score_str = f"{avg_clip_score:.3f}" if config.generate_images else "nan"
        file.write(
            f"BoolQ,{config.k if config.generate_images else 0},{config.batch_size},{config.num_test_samples},{avg_clip_score_str},{accuracy:.3f}\n"
        )

    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()

