import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm


def load_slimpajama(max_examples: int = 1000):
    def get_conversation(x):
        conversation = [
            {"role": "user", "content": ''},
            {"role": "assistant", "content": x["text"]}
        ]
        return {"conversation": conversation}

    dataset = load_dataset("DKYoon/SlimPajama-6B", split="validation")
    dataset = dataset.take(n=max_examples)
    dataset = dataset.map(get_conversation)

    conversations = list(dataset['conversation'])

    return conversations


def load_lmsys(max_examples: int = 1000):
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
    dataset = dataset.filter(lambda x: x['turn'] == 1)
    dataset = dataset.take(n=max_examples)
    
    conversations = list(dataset['conversation'])

    return conversations


def load_hex_phi(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 64, 
    device: str = 'cpu'
):
    """Load HEx-PHI dataset and generate refusal responses."""
    repo_id = "LLM-Tuning-Safety/HEx-PHI"

    category_files = [
        "category_1.csv",
        # "category_2.csv",  # Bad content
        "category_3.csv",
        "category_4.csv", 
        "category_5.csv",
        "category_6.csv",
        "category_7.csv",
        "category_8.csv",
        "category_9.csv",
        "category_10.csv",
        "category_11.csv"
    ]

    all_prompts = []

    print("Downloading HEx-PHI dataset CSV files...")

    for category_file in category_files:
        local_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=category_file,
            repo_type="dataset"
        )
        
        df = pd.read_csv(local_file_path)
        category_prompts = [df.columns[0]] + df[df.columns[0]].values.tolist()
        all_prompts.extend(category_prompts)

    conversations = []
    for user_prompt in tqdm(all_prompts, desc="Generating responses"):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens).squeeze(0)
        response_ids = outputs[input_ids.size(1):]

        if response_ids[0] != 235285: # Not 'I' - doesn't refuse; skip
            continue

        model_response = tokenizer.decode(response_ids)
        conversations.append([
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": model_response}
        ])

    print("Finished!")
    
    return conversations
