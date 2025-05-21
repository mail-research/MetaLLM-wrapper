# Get answers from LLMS on the MMLU dataset

import time
import re
import os
import csv
import torch
import pandas as pd
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import datetime

# Add at the top of your script, before imports
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# Precompiled regex for fast extraction of answer option (A, B, C, or D)
ANSWER_PATTERN = re.compile(r'\b([A-D])\b')

def build_prompt(example):
    """
    Build a prompt from an MMLU example.
    Expects example to have:
      - "question": the question text
      - "choices": either a list or a dict of answer choices
      - "answer": the correct option letter (e.g. "A")
    """
    prompt = f"Answer the following multiple-choice question with only the answer letter (A, B, C, or D):\nQuestion: {example['question']}\nOptions:\n"
    
    if isinstance(example["choices"], list):
        for i, choice in enumerate(example["choices"]):
            letter = chr(65 + i)  # A, B, C, D...
            prompt += f"{letter}. {choice.strip()}\n"
    elif isinstance(example["choices"], dict):
        for key, value in example["choices"].items():
            prompt += f"{key}. {value.strip()}\n"
    else:
        choices = example.get("choices", [])
        for i in range(4):
            key = chr(65 + i)
            choice = choices[i] if isinstance(choices, list) and i < len(choices) else f"Choice {i+1}"
            prompt += f"{key}. {choice}\n"
    
    prompt += "Answer:"
    return prompt

def parse_output(text):
    """
    Quickly extract the first occurrence of one of the answer letters.
    """
    match = ANSWER_PATTERN.search(text)
    return match.group(1) if match else None

def build_prompts(examples):
    return [build_prompt(example) for example in examples]

def evaluate_model(model_id, csv_data, subset_ratio=1.0, batch_size=8):
    """
    Evaluate a single model using the Transformers pipeline with proper device management.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Initialize tokenizer with default settings
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        raise
    
    # Explicitly set device and handle device mapping
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Special handling for Gemma model
        if "gemma" in model_id.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="sequential",  # Use sequential for Gemma
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            # Create pipeline with sequential device mapping for Gemma
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="sequential",  # Use sequential for Gemma
                trust_remote_code=True,
            )
        else:
            # Default handling for other models
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="balanced",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="balanced",
                trust_remote_code=True,
            )
    except Exception as e:
        print(f"Error loading model or creating pipeline: {e}")
        raise

    # Apply subset_ratio to sample a portion of the data
    if subset_ratio < 1.0:
        csv_data = csv_data.sample(frac=subset_ratio, random_state=42)
        print(f"Using subset of data: {len(csv_data)} examples ({subset_ratio:.2%} of original)")
    
    prompts = csv_data['prompt'].tolist()
    answers = csv_data['answer'].tolist()
    
    from datasets import Dataset
    prompt_dataset = Dataset.from_dict({"prompt": prompts})

    start_time = time.time()
    print(f"Generating responses for {len(prompts)} examples...")
    
    try:
        # Add error handling around generation
        generation_outputs = pipe(
            prompt_dataset['prompt'],
            max_new_tokens=20,
            do_sample=False,
            temperature=0.0,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        raise

    total_time = time.time() - start_time
    predictions = []

    correct = 0
    for output, prompt, answer in zip(generation_outputs, prompts, answers):
        try:
            generated_text = output[0]['generated_text']
            new_text = generated_text[len(prompt):]
            predicted = parse_output(new_text)

            if predicted == chr(answer + 65):
                correct += 1
            
            predictions.append({
                "prompt": prompt,
                "ground_truth": chr(answer + 65),
                "prediction": predicted if predicted else "NONE",
            })
        except Exception as e:
            print(f"Error processing prediction: {e}")
            predictions.append({
                "prompt": prompt,
                "ground_truth": chr(answer + 65),
                "prediction": "ERROR",
            })
            
    total = len(prompts)
    if total == 0:
        raise Exception("No examples processed. Check your model and GPU configuration.")
        
    accuracy = correct / total
    avg_time = total_time / total
    
    # Clean up CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return accuracy, avg_time, predictions


def get_model_id(model_name):
    model_lst = model_name.split(',')
    model_id_lst = []

    for model_name in model_lst:
        if model_name == 'gemma 2B':
            model_id_lst.append("google/gemma-2-2b-it")
        elif model_name == 'llama 3B':
            model_id_lst.append("meta-llama/Llama-3.2-3B-Instruct")
        elif model_name == "qwen 7B":
            model_id_lst.append("Qwen/Qwen2.5-7B-Instruct-1M")
        elif model_name == "mistral 7B":
            model_id_lst.append("mistralai/Mistral-7B-Instruct-v0.3")
        elif model_name == "phi 2B":
            model_id_lst.append("microsoft/phi-2")
        elif model_name == 'llama 8B':
            model_id_lst.append("meta-llama/Llama-3.1-8B-Instruct")
        elif model_name == "qwen 1.5B":
            model_id_lst.append("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        else:
            raise ValueError(f"Model {model_name} not supported")
    return model_id_lst

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama 3B,gemma 2B")
    parser.add_argument("--split", type=str, default="test", choices=["auxiliary_train", "test"], 
                        help="Dataset split to use for evaluation (auxiliary_train or test)")
    parser.add_argument("--devices", type=str, default="2,3", help="Devices to use for evaluation")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Subset ratio to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size to use for evaluation")
    return parser.parse_args()
    

if __name__ == "__main__":
    args = ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    os.makedirs("results+", exist_ok=True)
    
    print(f"Starting benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using {args.split} split for evaluation")
    
    # List of MMLU tasks
    MMLU_TASKS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", 
        "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", 
        "college_medicine", "college_physics", "computer_security", "conceptual_physics", 
        "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic", 
        "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science", 
        "high_school_european_history", "high_school_geography", "high_school_government_and_politics", 
        "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", 
        "high_school_physics", "high_school_psychology", "high_school_statistics", 
        "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality", 
        "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", 
        "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", 
        "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law", 
        "professional_medicine", "professional_psychology", "public_relations", "security_studies", 
        "sociology", "us_foreign_policy", "virology", "world_religions"
    ]
    
    model_id_lst = get_model_id(args.model)
    results = {}
    
    # Create a summary CSV file for each model (filename format: "{model_name}_{split}.csv")
    model_summary_files = {}
    model_summary_writers = {}
    for model_id in model_id_lst:
        model_name = model_id.split('/')[-1]
        summary_file_path = f"results+/{model_name}_{args.split}.csv"
        model_summary_files[model_name] = open(summary_file_path, 'w', newline='', encoding='utf-8')
        if args.split == 'test':
            model_summary_writers[model_name] = csv.DictWriter(
                model_summary_files[model_name], fieldnames=['task', 'accuracy', 'avg_time_sec']
            )
        else:
            model_summary_writers[model_name] = csv.DictWriter(
                model_summary_files[model_name], fieldnames=['accuracy', 'avg_time_sec']
            )
        model_summary_writers[model_name].writeheader()
    
    # Evaluate each model on each task and write the summary row to its file
    for model_id in model_id_lst:
        model_name = model_id.split('/')[-1]
        predictions_file_path = f"results+/{model_name}_{args.split}_predictions.csv"
        
        # For test split, create new file at the beginning to avoid appending to old runs
        if args.split == 'test':
            # Initialize the predictions file with headers for a fresh start
            with open(predictions_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['task', 'prompt', 'ground_truth', 'prediction'])
                writer.writeheader()
                
            for task in MMLU_TASKS:
                print(f"\nEvaluating model {model_name} (ID: {model_id}) on task {task}, {args.split} split...")
                try:
                    csv_data = pd.read_csv(f"mmlu_prompts/{args.split}/{task}_prompts.csv")
                    accuracy, avg_time, predictions = evaluate_model(model_id, csv_data, args.subset_ratio, args.batch_size)
                    results_key = f"{model_name}_{task}_{args.split}"
                    results[results_key] = {"accuracy": accuracy, "avg_time_sec": avg_time}
                    
                    model_summary_writers[model_name].writerow({
                        'task': task,
                        'accuracy': f"{accuracy * 100:.2f}%",
                        'avg_time_sec': f"{avg_time:.2f}"
                    })
                    model_summary_files[model_name].flush()
                    print(f"{model_name} on {task}: Accuracy: {accuracy * 100:.2f}%, Avg time: {avg_time:.2f} sec")

                    # Now always append to the file (header is already written)
                    with open(predictions_file_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=['task', 'prompt', 'ground_truth', 'prediction'])
                        # No longer need to check for first task since header is written separately
                        task_predictions = [{**p, 'task': task} for p in predictions]
                        writer.writerows(task_predictions)
                except Exception as e:
                    print(f"Error processing task {task}: {e}")
                    # Log the error in results
                    results[f"{model_name}_{task}_{args.split}"] = {"error": str(e)}
        else:
            print(f"Evaluating model {model_name} (ID: {model_id}) on {args.split} split...")
            csv_data = pd.read_csv(f"mmlu_prompts/{args.split}/prompts.csv")
            accuracy, avg_time, predictions = evaluate_model(model_id, csv_data, args.subset_ratio, args.batch_size)
            results_key = f"{model_name}_{args.split}"
            results[results_key] = {"accuracy": accuracy, "avg_time_sec": avg_time}

            model_summary_writers[model_name].writerow({
                'accuracy': f"{accuracy * 100:.2f}%",
                'avg_time_sec': f"{avg_time:.2f}"
            })
            model_summary_files[model_name].flush()
            print(f"{model_name} on {args.split} split: Accuracy: {accuracy * 100:.2f}%, Avg time: {avg_time:.2f} sec")
            
            # For other splits, write predictions directly
            with open(predictions_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['prompt', 'ground_truth', 'prediction'])
                writer.writeheader()
                writer.writerows(predictions)
    

    # Close all summary CSV files
    for file_handle in model_summary_files.values():
        file_handle.close()
        
    print("\nBenchmark Results:")
    for label, metrics in results.items():
        if "accuracy" in metrics:
            print(f"{label}: Accuracy: {metrics['accuracy'] * 100:.2f}%, Avg time: {metrics['avg_time_sec']:.2f} sec")
        else:
            print(f"{label}: ERROR - {metrics.get('error', 'Unknown error')}")