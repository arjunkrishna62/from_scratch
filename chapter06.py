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
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})) # for creating data_loader

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

print(train_dataset.max_length)

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

print(balanced_df["Label"].isna().sum())  # should be 0