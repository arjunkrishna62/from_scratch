# Preparing a dataset for supervised instruction fine-tuning

import json
import os
import urllib

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else: # skips dwld if file was alreday downloaded
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
            data = json.load(file)
    return data

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
data = download_and_load_file(file_path, url)
# print("Number of entries:", len(data))  # data list that we loaded from the JSON file contains the 1,100 entries of the instruction dataset

# fn to convert the data list into alpaca prompt style
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
# print(model_input + desired_response)

# dividing the dataset into train, test, val

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]


# print("Training set length:", len(train_data)) # 935
# print("Validation set length:", len(val_data)) # 55
# print("Test set length:", len(test_data)) # 110



# creating our own custom collate, later we will plug into data loaders
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data: # pretokenize data
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    def __getitem__(self, index):
        return self.encoded_texts[index]
    def __len__(self):
        return len(self.data)
    
# we use tekonizer to create padding tokens
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
# print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})) # 50256

# padding 
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch) # Finds the longest sequence in the batch
    inputs_lst = []
    for item in batch: # Pads and prepares inputs
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # removes the extra padded token added earlier
        inputs_lst.append(inputs)
        inputs_tensor = torch.stack(inputs_lst).to(device) # converts yhe list of integers into tensors and transfers it to the target device
    return inputs_tensor

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)
# print(custom_collate_draft_1(batch))
# tensor([[    0,     1,     2,     3,     4],
#         [    5,     6, 50256, 50256, 50256],
#         [    7,     8,     9, 50256, 50256]])

# The input and target token alignment used in the instruction fine-tuning process of an LLM. For each input sequence, the corresponding
# target sequence is created by shifting the token IDs one position to the right, omitting the first token of the input, and appending an end-of-text token.

def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # truncates last token for input
        targets = torch.tensor(padded[1:]) # shift +1 for the right of targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
inputs, targets = custom_collate_draft_2(batch)
# print('inputs',inputs)
# print('targets after shifted by +1 pos:\n',targets)

# Replace certain padding tokens by -100 to exclude them from the training loss.

# replace tokens with ID 50256 with -100 in the target lists. Additionally, we introduce an allowed_max_length 
# parameter to optionally limit the length of the samples.

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_fn(batch)
# print('inputs:\n',inputs)
# print('padded targets after replacing the token id 50256 with -100: \n',targets)


# Creating data loaders for an instruction dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# print("Device:", device) # cpu

from functools import partial
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

# initializing data loders - automaticallly shuffle and oragnize the batches for llm instruction for fine tuing process

from torch.utils.data import DataLoader

num_workers = 0
batch_size = 2
torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# This output shows that the first input and target batch have dimensions 8 × 61, where 8 represents the batch size and 61 is the number of tokens in each training example in
# this batch. The second input and target batch have a different number of tokens—for instance, 76. Thanks to our custom collate function, the data loader is able to create
# batches of different lengths

# print("Train loader:")
# for inputs, targets in train_loader:
#     print(inputs.shape, targets.shape)

# we load a pre-trained LLM to fine-tune with this data loader

# Loading a pretrained LLM

from gpt_download import download_and_load_gpt2
from chapter04 import GPTModel
from chapter05 import load_weights_into_gpt

BASE_CONFIG = {
    "vocab_size": 50257, # vocabulary size
    "context_length": 1024, # context length
    "drop_rate": 0.0, # dropout rate
    "qkv_bias": True # query-key-value bias
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

torch.manual_seed(123)
input_text = format_input(val_data[0])
# print(input_text)

from chapter05 import generate, text_to_token_ids, token_ids_to_text

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)

response_text = generated_text[len(input_text):].strip()
print(response_text) # this output shows that the pretrained model is not yet capable of correctly following the given instruction.

# fine tuning the llm on instruction data

from chapter05 import (
    calc_loss_loader,
    train_model_simple
)

model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(
    train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
)
# print("Training loss:", train_loss) # 3.825908660888672
# print("Validation loss:", val_loss) # 3.7619335651397705

# Instruction fine-tuning the pretrained LLM

# import time

# start_time = time.time()
# torch.manual_seed(123)
# optimizer = torch.optim.AdamW(
#     model.parameters(), lr=0.00005, weight_decay=0.1
# )
# num_epochs = 2
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=55, eval_iter=55,
#     start_context=format_input(val_data[0]), tokenizer=tokenizer
# )
# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.") # we can see that model is learning graudually , consistenttly decreasing training and val loss over thw two epochs
# on CPU it will take 15-16 mins for 2 epochs while on GPU A100 0.86sec seconds or with L4 - 1.83 seconds. 

# from chapter05 import plot_losses
# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# torch.manual_seed(123) 
# for entry in test_data[:3]: # Iterates over the first three test set samples
#     input_text = format_input(entry)
#     token_ids = generate( # Uses the generate function imported in section 7.5
#         model=model,
#         idx=text_to_token_ids(input_text, tokenizer).to(device),
#         max_new_tokens=256,
#         context_size=BASE_CONFIG["context_length"],
#         eos_id=50256
#     )
#     generated_text = token_ids_to_text(token_ids, tokenizer)
#     response_text = (
#         generated_text[len(input_text):]
#         .replace("### Response:", "")
#         .strip()
#     )

# print(input_text)
# print(f"\nCorrect response:\n>> {entry['output']}")
# print(f"\nModel response:\n>> {response_text.strip()}")
# print("-------------------------------------")

# generating the test set respones (evaluating the model perfomance)

from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text
with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)

#  evaluating that ollama is working 

import psutil
def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")
if not ollama_running:
    raise RuntimeError(
        "Ollama not running. Launch ollama before proceeding."
    )
print("Ollama running:", check_if_running("ollama")) # displays ollama running : True

# Querying a local Ollama model

import urllib.request
def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = {
        "model": model, # creates the data payload as a dictionary
        "messages": [
            {"role": "user", "content": prompt}
    ],
    "options": { # setting for deterministic responses
        "seed": 123,
        "temperature": 0,
    "num_ctx": 2048
        }
    }

    payload = json.dumps(data).encode("utf-8")  # converts the dictionary to a JSON formatted string and encodes it to bytes
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )

    request.add_header("Content-Type", "application/json") # creates a req object, setting the method to POST and adding neccassry headers

    response_data = ""
    with urllib.request.urlopen(request) as response: # sends the req and captures the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data

model = "llama3"
result = query_model("What do Llamas eat?", model)
print(result)

# using the query_model function defined earlier, we can evaluate the responses generated by our fine-tuned model that prompts the llama 3 model to rate our fine tuned
# model’s responses on a scale from 0 to 100 based on the given test set response as reference.

for entry in test_data[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
print("\nDataset response:")
print(">>", entry['output'])
print("\nModel response:")
print(">>", entry["model_response"])
print("\nScore:")
print(">>", query_model(prompt))
print("\n-------------------------")

# evaluating instruction fine tuning llm
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."  # Modifiedinstruction line to only return the score
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue
    return scores


scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")