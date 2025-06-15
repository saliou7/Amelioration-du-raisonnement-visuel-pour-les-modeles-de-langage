import os
import random
import sys
import torch
import numpy as np
from tqdm.auto import tqdm
from nltk import sent_tokenize
from torch.utils.data import DataLoader
from collections import Counter
import re
import string

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from datasets import load_dataset
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, AutoConfig
from diffusers import AutoPipelineForText2Image

from nltk.corpus import stopwords

# Download required NLTK data
import nltk
nltk.data.path.append("/home/saliou/nltk_data")
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Load stopwords for English
stop_words = set(stopwords.words("english"))


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.testing_config import parse_args
from inference import load_checkpoint, compute_scores


def tokenize_and_clean(text):
    # Remove punctuation and split into words
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    return words


def calculate_f1(pred, truth):
    # Clean and tokenize both prediction and truth
    pred_tokens = tokenize_and_clean(pred)
    truth_tokens = tokenize_and_clean(truth)

    # print("#######################################################################")
    # print("Truth tokens:", truth_tokens)
    # print("Pred tokens:", pred_tokens)
    # print("#######################################################################")

    # Compute overlaps
    overlap = Counter(pred_tokens) & Counter(truth_tokens)
    num_matches = sum(overlap.values())

    if num_matches == 0:
        return 0.0

    # Calculate precision and recall
    precision = num_matches / len(pred_tokens) if pred_tokens else 0
    recall = num_matches / len(truth_tokens) if truth_tokens else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def is_exact_match(pred, truth):
    # Normalize and clean both prediction and truth
    pred_tokens = ' '.join(tokenize_and_clean(pred))
    truth_tokens = ' '.join(tokenize_and_clean(truth))
    return pred_tokens == truth_tokens


def normalize(text):
    return ' '.join(word.lower() for word in text.split())


# def extract_answer(full_response):
#     match = re.search(r'SHORT ANSWER:\s*([^\n]+)', full_response)
#     if match:
#         return match.group(1).strip()
#     # Fallback to extracting the first line if no "SHORT ANSWER:"
#     return full_response.strip().split("\n")[0] or "I don't know"

def extract_answer(full_response):
    lines = full_response.strip().split("\n")
    for line in lines:
        if "SHORT ANSWER:" in line:
            return line.split("SHORT ANSWER:")[-1].strip()
    return lines[0].strip()  # Fallback to first line


def main():
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer=load_checkpoint(config.path)

    # Image generation setup
    t2i_model = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device) if config.generate_images else None

    # Load the CLIP model for alignment scoring
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = clip_processor.tokenizer # Get the tokenizer

    # Dataset setup
    squad_data = load_dataset("squad_v2")
    val_data = squad_data["validation"].select(range(config.num_test_samples))

    cumulative_f1 = cumulative_exact = total_samples = 0
    inst = 0
    
    # Few-shot tracking variables
    prvs_title = None
    prvs_context = None
    prvs_question = None
    prvs_answer = None
    prvs_flag = False
    scores = None
    avg_clip_scores = []

    with torch.inference_mode():
        for entry in tqdm(val_data, desc="Evaluating SQuAD 2.0"):
            current_title = entry['title']
            current_context = entry['context']
            current_question = entry['question']
            current_answers = entry['answers']['text']
            inst += 1

            if not current_answers:
                continue

            # Handle title changes and few-shot examples
            if current_title != prvs_title:
                # New title - store first example and skip processing
                prvs_title = current_title
                prvs_context = current_context
                prvs_question = current_question
                prvs_answer = current_answers[0]
                skip_flag = True
            else:
                skip_flag = False

            if skip_flag:
                print(f"Skipping first item of title: {current_title}")
                skip_flag = False
                continue


            input_prompt = (
            "Answer the question concisely using examples from the text.\n"
            "Make sure to include 'SHORT ANSWER:' before your answer.\n\n"
            f"TITLE: {prvs_title}\n"
            # f"EXAMPLE TEXT: {prvs_context}\n"
            f"EXAMPLE QUESTION: {prvs_question}\n"
            f"EXAMPLE ANSWER: {prvs_answer}\n\n"
            f"TEXT: {current_context}\n"
            f"QUESTION: {current_question}\n"
            "SHORT ANSWER:"
            )
            


            # Tokenization
            inputs = tokenizer(
                input_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            ).to(device)

            # Image generation
            pixel_values = None
            if config.generate_images and t2i_model:
                # Split CURRENT context into sentences
                current_prompt = f"{current_context} {current_question}"  # Combine text
                prompt_sentences = sent_tokenize(current_prompt.strip())
                
                # Handle sentence sampling
                if len(prompt_sentences) >= config.k:
                    selected_sentences = random.sample(prompt_sentences, config.k)
                else:
                    # Random sampling with replacement if needed
                    selected_sentences = [random.choice(prompt_sentences) for _ in range(config.k)]

                print(selected_sentences)
                
                # Generate 1 image per sentence
                image_prompts = [f"Visual context: {sent}" for sent in selected_sentences]
                
                images = t2i_model(image_prompts, num_inference_steps=25,guidance_scale=7.5, generator=torch.manual_seed(42)).images

                scores = compute_scores(clip_model, clip_processor, images, selected_sentences) 
                avg_clip_scores.append(scores.mean().item())
                print("Scores:", scores)
                print("Score:", avg_clip_scores[-1])


                pixel_values = model.processor(images=images, return_tensors="pt").pixel_values.to(device)


                os.makedirs("output/images/squad", exist_ok=True)
                if config.show_images:
                    for i, image in enumerate(images):
                        image.show()
                        if config.save_images:
                            image.save(f"output/images/squad{inst}_{i}.png")
            


            # Text generation
            output_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                scores=scores,
                pixel_values=pixel_values,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=config.max_new_tokens, 

            )

            # Extract and process answer
            full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # print("Full response:", full_response)  # Debugging

            full_response = full_response[len(input_prompt):].strip().split('.')[0]
            prediction = extract_answer(full_response)


            best_exact = 0
            for gt in current_answers:
                best_exact = max(best_exact, is_exact_match(prediction, gt))
            cumulative_exact += best_exact

            # Compute best F1 score
            best_f1 = 0
            for gt in current_answers:
                best_f1 = max(best_f1, calculate_f1(prediction, gt))
            cumulative_f1 += best_f1

            total_samples += 1

            # Update prvs example with current context
            prvs_context = current_context
            prvs_question = current_question
            prvs_answer = current_answers[0]

            # print("#######################################################################")
            print(f"Truth: {current_answers} ")
            print(f"Prediction: {prediction} ")
            print(f"Current F1: {best_f1}")
            print(f"Current cum F1: {cumulative_f1/total_samples:.3f}")
            print("#######################################################################")

    # Final results and saving
    final_exact = 100.0 * (cumulative_exact / total_samples)
    final_f1 = 100.0 * (cumulative_f1 / total_samples)
    print(f"Final Results - Exact Match: {final_exact:.2f}%, F1: {final_f1:.2f}%")
    if config.generate_images:
        avg_clip_score = np.mean(avg_clip_scores)
        print(f"Average CLIP score: {avg_clip_score:.3f}")

    results_path = "output/results/squad_evaluation.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    

    with open(results_path, 'a') as file:
        if not os.path.exists(results_path):
            file.write("Dataset,FewShots,K,Batch Size,Num Test Samples,max_new_tokens,Exact Match,Avg Clip score,F1\n")

        
        avg_clip_score_str = f"{avg_clip_score:.3f}" if config.generate_images else "nan"
        file.write(
            f"SQuAD 2.0,YES,{config.k if config.generate_images else 0},{config.batch_size},{config.num_test_samples},{config.max_new_tokens},{final_exact:.3f},{avg_clip_score_str},{final_f1:.3f}\n"
        )

if __name__ == "__main__":
    main()