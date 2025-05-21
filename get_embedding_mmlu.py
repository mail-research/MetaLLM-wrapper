import torch
import pandas as pd
import tqdm
import os
import ast
import argparse

from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

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

class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]
    
@torch.no_grad()
def get_feat_mpnet(model, prompts, tokenizer, device, batch_size=64):
    # Create a DataLoader for batch processing
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f'Getting embeddings for {len(prompts)} prompts')
    model.to(device).eval()
    all_embeddings = []
    
    for batch in tqdm.tqdm(dataloader):
        batch_dict = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        input_ids = batch_dict['input_ids'].to(device)
        attention_mask = batch_dict['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        all_embeddings.append(model.pooler(outputs['hidden_states'][-1]))

    # Concatenate all batch embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

@torch.no_grad()
def get_feat_e5(model, prompts, tokenizer, device, batch_size=32):
    # Create a DataLoader for batch processing
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f'Getting embeddings for {len(prompts)} prompts')
    model.eval()
    all_embeddings = []
    
    for batch in tqdm.tqdm(dataloader):
        batch_dict = tokenizer(batch, padding=True, max_length=4096, truncation=True, return_tensors='pt')        
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        all_embeddings.append(embeddings.cpu())
    
    # Concatenate all batch embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
def get_detailed_instruct(task, questions, choices, split) -> list:
    prompts = []
    for question, option_list in zip(questions, choices):
        # Convert task name from snake_case to space-separated words
        task_name = ' '.join(task.split('_'))
        
        # Letter labels for the options
        letter_labels = ['A', 'B', 'C', 'D']
        
        # Build the choices string with each option on a new line
        choice_str = ''
        for i, option in enumerate(ast.literal_eval(option_list)):
            choice_str += f'{letter_labels[i]}). {option}\n'

        # Construct the complete prompt
        prompt = (
            f"Instruct: Given a multiple-choice question on {task_name}, select the most correct answer from the choices.\n"
            f"Query: {question}\n"
            f"Choices:\n{choice_str}"
        )
        prompts.append(prompt)
    return prompts

if __name__ == '__main__':
    # We can incorporate task name for auxiliary_split from https://huggingface.co/datasets/kz919/mmlu-auxiliary-train-auto-labelled
    
    # Add argument parser
    parser = argparse.ArgumentParser(description='Generate embeddings from text using a pre-trained model')
    parser.add_argument('--model', type=str, default='intfloat/e5-mistral-7b-instruct', choices=['sentence-transformers/all-mpnet-base-v2', 'intfloat/e5-mistral-7b-instruct'],
                        help='Hugging Face model name/path to use for embeddings')
    parser.add_argument('--split', type=str, default='auxiliary_train', choices=['auxiliary_train', 'test'],
                        help='Split to use for embeddings')
    parser.add_argument('--devices', type=str, default='0,1,2')
    parser.add_argument('--miscellaneous_train_part', type=int, default=None) # Split the miscellaneous train set into 4 parts
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # Use model name from arguments
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, device_map='auto')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    all_embeddings = []
    
    model_name = args.model.split('/')[-1]
    for task in MMLU_TASKS:
        if args.split == 'auxiliary_train':
            split = 'train'
        elif args.split == 'test':
            split = 'test'

        os.makedirs(f'mmlu_tasks_split/{task}/{model_name}_embeddings', exist_ok=True)

        if task == 'miscellaneous' and args.miscellaneous_train_part is not None:
            if args.split != 'auxiliary_train':
                raise ValueError('miscellaneous_train_part is only supported for auxiliary_train split')
            embedding_path = f'mmlu_tasks_split/{task}/{model_name}_embeddings/{args.split}_{args.miscellaneous_train_part}.pt'
        else:
            embedding_path = f'mmlu_tasks_split/{task}/{model_name}_embeddings/{args.split}.pt'

        print("Embedding path: ", embedding_path)

        if os.path.exists(embedding_path):
            print(f'Skipping {task} {split} because it already exists')
            # all_embeddings.append(torch.load(embedding_path).cpu())
            continue

        data = pd.read_csv(f'mmlu_tasks_split/{task}/{split}.csv')
        if len(data) == 0:
            continue

        questions = data['question'].tolist()
        choices = data['choices'].tolist()
        prompts = get_detailed_instruct(task, questions, choices, split)
        
        if args.miscellaneous_train_part is not None:
            split_size = len(prompts) // 4
            if args.miscellaneous_train_part == 4:
                prompts = prompts[(args.miscellaneous_train_part-1) * split_size:]
            else:
                prompts = prompts[(args.miscellaneous_train_part-1) * split_size: args.miscellaneous_train_part * split_size]

        if args.model == 'sentence-transformers/all-mpnet-base-v2':
            feats = get_feat_mpnet(model, prompts, tokenizer, device, batch_size=64)
        else:
            feats = get_feat_e5(model, prompts, tokenizer, device, batch_size=4)

        torch.save(feats, embedding_path)
        all_embeddings.append(feats.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(all_embeddings, f'mmlu_tasks_split/all/{args.model}_embeddings/{args.split}.pt')
    
    