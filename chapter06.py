import urllib.request
import zipfile
import os
from pathlib import Path
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download " "and extraction." )
        return
    with urllib.request.urlopen(url) as response: # dwnlds the file
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref: # unzip the file
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path) # adds a .tsv file extention
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

import pandas as pd
df = pd.read_csv(
data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
print(df)
print(df["Label"].value_counts()) # ham - not spam

# We can use the code in the following listing to undersample and create a balanced  dataset.
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0] # counts the instances of "spam"
    ham_subset = df[df["Label"] == "ham"].sample( 
        num_spam, random_state=123 # randomly smples "ham" instances to match the number of "spam" instances
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"] # combine ham subset wth "spam"
    ])
    return balanced_df
balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

# convert the “string” class labels "ham" and "spam" into integer class labels 0 and 1
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# splittting the dataset
def random_split(df, train_frac, validation_frac):
    
    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True) # shuffles the entire df
    train_end = int(len(df) * train_frac) # clacts split indices
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end] # splits the df
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    
    return train_df, validation_df, test_df
    
train_df, validation_df, test_df = random_split( balanced_df, 0.7, 0.1 ) # test size is implied to be 0.2 as the remainder

train_df.to_csv("train.csv", index=None) # save , for reusability
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
# print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})) # for creating data_loader we use speciliased to proprocess text  (it will print [50256])

# creating data loaders (like in text, each batch(chunk) for individual training unit
# working with a spam dataset that contains text messages of varying lengths. To batch these messages as we did with the text chunks, we have two primary options:
# -> Truncate all messages to the length of the shortest message in the dataset or batch (computationlly cheaper but significant info loss)
# -> Pad all messages to the length of the longest message in the dataset or batch (gooing with this, padded to the longest msg in the seq)

import torch
from torch.utils.data import Dataset
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [ # Pretokenizes texts
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            
            self.encoded_texts = [  # Truncates sequences if they are longer than max_length
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
            
        self.encoded_texts = [ # Pads sequences to the longest sequence
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
            
    def __getitem__(self, index):
        
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        
        # print(f"[DEBUG] index={index}, label={label}, type={type(label)}")

        # try:
        #     label_tensor = torch.tensor(int(label), dtype=torch.long)
        # except Exception as e:
        #     print(f"[ERROR] Failed to convert label at index={index}")
        #     print(f"Label value: {label} (type: {type(label)})")
        #     raise e
            
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
        
    def __len__(self):
            return len(self.data)
            
    def _longest_encoded_length(self):
            
            max_length = 0
            for encoded_text in self.encoded_texts:
                encoded_length = len(encoded_text)
                if encoded_length > max_length:
                    max_length = encoded_length
            return max_length

train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)

# print(train_dataset.max_length) # should print 120

# we pad the validation and test sets to match the length of the longest training sequence.
# Importantly, any validation and test set samples exceeding the length of the longest training example are truncated using encoded_text[:self.max_length] in the SpamDataset.
# This truncation is optional; you can set max_length=None for both validation and test sets, provided there are no sequences exceeding 1,024 tokens in these sets:

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

# creating data loaders
from torch.utils.data import DataLoader
    
num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

# print(balanced_df["Label"].isna().sum())  # should be 0

# for input_batch, target_batch in train_loader: # to ensure tha data loader is working
    # pass
# print("Input batch dimensions:", input_batch.shape) # 8, 120
# print("Label batch dimensions", target_batch.shape) # 8

# print(f"{len(train_loader)} training batches") # 130 training batches
# print(f"{len(val_loader)} validation batches") # 19 val batches
# print(f"{len(test_loader)} test batches") # 38 test batches

# initializing a model with pretrained weights

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
    
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Loading a pretrained GPT model

from gpt_download import download_and_load_gpt2
from chapter05 import GPTModel, load_weights_into_gpt

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# After loading the model weights into the GPTModel, we reuse the text generation utility function from chapters 4 and 5 to ensure that the model generates coherent text:
from chapter04 import generate_text_simple
from chapter05 import text_to_token_ids, token_ids_to_text
text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

# Before we start fine-tuning the model as a spam classifier, let’s see whether the model already classifies spam messages by prompting it with instructions:
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))

# Adding a classification head
# Fine-tuning selected layers vs. all layers
# Since we start with a pretrained model, it’s not necessary to fine-tune all model layers.
# In neural network-based language models, the lower layers generally capture basic language structures and semantics applicable across a wide range of tasks and datasets.
# So, fine-tuning only the last layers (i.e., layers near the output), which are more specific to nuanced linguistic patterns and task-specific features, is often sufficient to adapt the
# model to new tasks. A nice side effect is that it is computationally more efficient to finetune only a small number of layers.

# To get the model ready for classification fine-tuning, we first freeze the model, meaning that we make all layers nontrainable:
for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

# To make the final LayerNorm and last transformer block trainable, we set their respective requires_grad to True:
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
# print("Inputs:", inputs) # tensor([[5211,  345,  423,  640]])
# print("Inputs dimensions:", inputs.shape) # torch.Size([1, 4])

with torch.no_grad():
    outputs = model(inputs)
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)

# Calculating the classification loss and accuracy by applying argmax based prediction code to all examples in the dataset

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    
    model.eval()
    correct_predictions, num_examples = 0, 0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :] # logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)
            
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
    return correct_predictions / num_examples
    
# function to determine the classification accuracies across various datasets estimated from 10 batches for efficiency:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)
val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)
test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)
print(f"Training accuracy: {train_accuracy*100:.2f}%") # 46.25%
print(f"Validation accuracy: {val_accuracy*100:.2f}%") # 45.00%
print(f"Test accuracy: {test_accuracy*100:.2f}%") # 48.75%
# prediction accuracies are near a random prediction
# objective is to maximize the spam classifier acccuracy of hte model


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :] # logits of last o/p token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# calc_loss_batch function to compute the loss for a single batch obtained from the previously defined data loaders. To calculate the loss for all batches in a data
# loader, we define the calc_loss_loader function as before.
def calc_loss_loader(data_loader, model, device, num_batches=None): # calcg the classification loss
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


with torch.no_grad(): # # Disables gradient tracking for efficiency because we are not training yet
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5 
    )
val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
print(f"Training loss: {train_loss:.3f}") # 2.453
print(f"Validation loss: {val_loss:.3f}") # 2.583
print(f"Test loss: {test_loss:.3f}") # 2.322

# Fine-tuning the model on supervised data

# implement a training function to fine-tune the model, which means adjusting the model to minimize the training set loss. Minimizing the training set loss
# will help increase the classification accuracy, which is the overall goal

def train_classifier_simple( model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], [] # to track loses and examples seen
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs): # main training loop
        model.train() # sets model to training mode

        for input_batch, target_batch in train_loader: 
            optimizer.zero_grad() # # resets the loss grad from the previous batch iteration
            loss = calc_loss_batch(
            input_batch, target_batch, model, device
            )
            loss.backward() # calc loss gradient
            optimizer.step() # updates models weights using loss gradients
            examples_seen += input_batch.shape[0] # new: tracks examples instead of tokens
            global_step += 1

            if global_step % eval_freq == 0: # optional eval step
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
                
        train_accuracy = calc_accuracy_loader( # Calculates accuracy after each epoch
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    return train_losses, val_losses, train_accs, val_accs, examples_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5
    )

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.") # Training accuracy: 100.00% | Validation accuracy: 97.50% Training completed in 5.65 minutes.

# calculate perfomance matrix
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%") # 97.21%
print(f"Validation accuracy: {val_accuracy*100:.2f}%") # 97.32%
print(f"Test accuracy: {test_accuracy*100:.2f}%") # 95.67%

# Using the LLM as a spam classifier using the model to classify new texts
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text) # prepare inputs to the model
    supported_context_length = model.pos_emb.weight.shape[1]

    input_ids = input_ids[:min( # truncates seqs if they are too long
    max_length, supported_context_length
    )]

    input_ids += [pad_token_id] * (max_length - len(input_ids)) # pads seqs to the longest seq

    input_tensor = torch.tensor( # adds batch dim
        input_ids, device=device
    ).unsqueeze(0) 

    with torch.no_grad(): # models inference without gradient tracking
        logits = model(input_tensor)[:, -1, :] # logits of the last o/p token
    predicted_label = torch.argmax(logits, dim=-1).item()
    
    return "spam" if predicted_label == 1 else "not spam" # last classified ouput

text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)
print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

torch.save(model.state_dict(), "review_classifier.pth")

model_state_dict = torch.load("review_classifier.pth", map_location=device)
model.load_state_dict(model_state_dict)

