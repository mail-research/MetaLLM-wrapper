import torch
import pandas as pd
import torch
import argparse

from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class LinUCB:
    def __init__(self, d, arms, prices, alpha=1.0, lambda_reg=1.0, gamma=1.0, p=0.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize LinUCB with a discount factor, using PyTorch and CUDA if available.
        
        Parameters:
        - d: int, dimension of context
        - alpha: float, exploration parameter
        - lambda_reg: float, regularization parameter
        - gamma: float, discount factor (0 < gamma <= 1), default 0.99
        - device: str, 'cuda' or 'cpu', defaults to 'cuda' if available
        """
        self.d = d
        self.arms = arms
        self.num_arms = len(arms)
        self.prices = prices
        
        self.prices = prices
        self.normalized_prices = {k: v / max(prices.values()) for k, v in prices.items()}

        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.device = torch.device(device)
        self.A = {}
        self.b = {}
        self.p = p

    def initialize_arm(self, arm, past_contexts=None, past_labels=None):
        """
        Initialize an arm with optional past data (no discounting applied here).
        
        Parameters:
        - arm: identifier for the arm (e.g., int or str)
        - past_contexts: torch.Tensor or None, past context vectors
        - past_labels: torch.Tensor or None, labels of past contexts for each model
        """
        # Initialize A as lambda_reg * identity matrix on the specified device
        self.A[arm] = self.lambda_reg * torch.eye(self.d, device=self.device)
        # Initialize b as zero vector on the specified device
        self.b[arm] = torch.zeros(self.d, device=self.device)
        
        if past_contexts is not None and past_labels is not None:
            print(f"Initializing arm {arm} with past contexts and labels")
            arm_idx = self.arms.index(arm)
            arm_labels = past_labels[range(len(past_labels)), arm_idx]
            past_rewards = self.get_rewards(arm=arm, corrects=arm_labels)

            for x, r in zip(past_contexts, past_rewards):
                # Ensure context is a tensor on the correct device
                x = torch.tensor(x, dtype=torch.float32, device=self.device) if not torch.is_tensor(x) else x.to(self.device)
                # Update A and b
                self.A[arm] += torch.outer(x, x)
                self.b[arm] += r * x

    def select_arm(self, context):
        """
        Select the arm with the highest upper confidence bound.
        
        Parameters:
        - context: torch.Tensor, context vector (x_t) of shape (d,)
        
        Returns:
        - best_arm: the selected arm identifier
        """
        # Ensure context is on the correct device
        # context = torch.tensor(context, dtype=torch.float32, device=self.device) if not torch.is_tensor(context) else context.to(self.device)
        
        best_value = float('-inf')
        best_arm = None
        for arm in self.arms:
            # Compute inverse of A (on GPU if device is 'cuda')
            A_inv = torch.linalg.inv(self.A[arm])
            # Compute theta = A_inv @ b
            theta = A_inv @ self.b[arm]
            # Compute UCB: theta^T x + alpha * sqrt(x^T A_inv x)
            mean = torch.dot(theta, context)
            uncertainty = self.alpha * torch.sqrt(torch.dot(context, A_inv @ context))
            p = mean + uncertainty
            
            if p > best_value:
                best_value = p.item()  # Convert to scalar for comparison
                best_arm = arm
        return best_arm

    def update(self, arm, context, reward):
        """
        Update A and b for the chosen arm with discounting.
        
        Parameters:
        - arm: selected arm identifier
        - context: torch.Tensor, context vector (x_t) of shape (d,)
        - reward: float, observed reward (r_t)
        """
        # Ensure context is on the correct device
        # context = torch.tensor(context, dtype=torch.float32, device=self.device) if not torch.is_tensor(context) else context.to(self.device)
        
        # Apply discount factor to existing A and b, then add new observation
        self.A[arm] = self.gamma * self.A[arm] + torch.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context


    def get_rewards(self, arm, corrects):
        return corrects.float() - self.p * self.normalized_prices[arm]

    def simulation(self, test_contexts, test_labels, update=True):
        print("Perfoming simulation...")
        policy = []
        policy_index = []
        accuracy = 0
        total_price = 0
        for i, context in enumerate(pbar := tqdm(test_contexts, disable=False)):
            # Select arm and update policy
            selected_arm = self.select_arm(context)
            selected_arm_index = self.arms.index(selected_arm)
            policy.append(selected_arm)
            policy_index.append(selected_arm_index)

            # Get reward and update arm's model
            correct = test_labels[i, selected_arm_index]
            reward = self.get_rewards(arm=selected_arm, corrects=correct)
            accuracy += correct.item()
            total_price += self.prices[selected_arm]
            if update:
                self.update(selected_arm, context, reward=reward)
            pbar.set_description(f'Test Accuracy: {accuracy / (i + 1):.4f}, Test Price: {total_price:.2f}')
        
        # Calculate accuracy and price
        chosen_arm_idx = torch.tensor(policy_index).cpu()
        correct = test_labels[range(len(test_labels)), chosen_arm_idx]
        accuracy = correct.sum().item() / len(test_labels)
    
        price
        counter = Counter(policy)
        total_price = 0.0
        for arm, count in counter.items():
            total_price += self.prices[arm] * count

        # Calculate price per 10000 samples
        total_price = total_price / len(test_contexts) * 10000
        
        print(f'Test Accuracy: {accuracy / len(test_contexts):.4f}, Test Price: {total_price:.2f}')
        return accuracy / len(test_contexts), total_price

def split_data(choices, random_state=1010) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompts_df = pd.read_csv('mmlu_prompts/test/all_mmlu_prompts.csv')
    test_data = pd.read_csv('mmlu_tasks_split/all/test.csv')
    test_data['task'] = prompts_df['task']

    test_embeddings = torch.load('mmlu_tasks_split/all/e5-mistral-7b-instruct_embeddings/test.pt')

    # Create an index array to track original positions
    test_data['original_index'] = range(len(test_data))

    # Perform stratified split
    train_idx, val_idx = train_test_split(
        test_data['original_index'],
        train_size=0.75,
        stratify=test_data['task'],
        random_state=1010
    )

    # Sort indices to preserve original order
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)

    # Split the data
    train_df = test_data.iloc[train_idx].drop('original_index', axis=1)
    val_df = test_data.iloc[val_idx].drop('original_index', axis=1)
    train_embeddings = test_embeddings[train_idx]
    val_embeddings = test_embeddings[val_idx]

    # Get labels
    train_labels = torch.tensor(train_df[choices].values).to(torch.float32)
    val_labels = torch.tensor(val_df[choices].values).to(torch.float32)

    return train_embeddings, val_embeddings, train_labels, val_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LinUCB Router for Language Models')
    parser.add_argument('--alpha', type=float, default=0, help='Exploration parameter')
    parser.add_argument('--lambda_reg', type=float, default=12.94, help='Regularization parameter')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount factor')
    parser.add_argument('--p', type=float, default=0.0, help='Price sensitivity parameter')
    parser.add_argument('--device', type=str, default='cuda:5', help='Device to run on (e.g., cuda:0, cpu)')
    parser.add_argument('--past_context', action='store_true', help='Whether to use past context')
    parser.add_argument('--update', action='store_true', help='Whether to update the model')
    args = parser.parse_args()

    choices = ['gemma', 'llama8B', 'mistral', 'qwen']
    prices = {'gemma': 0.1, 'llama8B': 0.18, 'mistral': 0.2, 'qwen': 0.3}

    # train_embeddings, val_embeddings, train_labels, val_labels = split_data(choices)
    train_embeddings = torch.load('mmlu_tasks_split/all/e5-mistral-7b-instruct_embeddings/auxiliary_train.pt').to('cpu')
    val_embeddings = torch.load('mmlu_tasks_split/all_filtered/e5-mistral-7b-instruct_embeddings/test.pt').to(args.device)

    train_data = pd.read_csv('mmlu_tasks_split/all/train.csv')
    val_data = pd.read_csv('mmlu_tasks_split/all_filtered/test.csv')

    train_labels = torch.tensor(train_data[choices].values.astype(int))
    val_labels = torch.tensor(val_data[choices].values.astype(int))

    total_accuracy = 0.0
    total_price = 0.0
    for i in range(3):
        bandit = LinUCB(
            d=4096, 
            arms=choices, 
            prices=prices, 
            alpha=args.alpha, 
            lambda_reg=args.lambda_reg, 
            gamma=args.gamma, 
            p=args.p, 
            device=args.device
        )

        for arm in choices:
            if args.past_context:
                bandit.initialize_arm(arm=arm, past_contexts=train_embeddings, past_labels=train_labels)
            else:
                bandit.initialize_arm(arm=arm, past_contexts=None, past_labels=None)
        
        # Shuffle embeddings and labels
        indices = torch.randperm(len(val_embeddings))
        val_embeddings = val_embeddings[indices]
        val_labels = val_labels[indices]
    
        accuracy, price = bandit.simulation(test_contexts=val_embeddings, test_labels=val_labels, update=False)
        total_accuracy += accuracy
        total_price += price

    print(f'Total Accuracy: {total_accuracy / 3:.4f}, Total Price: {total_price / 3:.2f}')