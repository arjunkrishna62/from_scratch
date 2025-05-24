# import urllib.request
# import re

# url = ("https://raw.githubusercontent.com/rasbt/"
# "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
# "the-verdict.txt")
# file_path = "the-verdict.txt"
# urllib.request.urlretrieve(url, file_path)

# with open("the-verdict.txt", "r", encoding="utf-8") as f:raw_text = f.read()
# print("Total number of character:", len(raw_text))
# print('verdict.txt raw out',raw_text[:99])

# import re
# text = "Hello, world. This, is a test."
# result = re.split(r'(\s)', text)
# print('tokenizing', result)

# result = re.split(r'([,.]|\s)', text)
# print('seperates commas, and dots',result)

# # result = [item for item in result if item.strip()]
# # print(result)

# ext = "Hello, world. Is this-- a test?"
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# print(result)
# result = [item.strip() for item in result if item.strip()]
# print('tokeninzing the text with special chars',result)

# # applying it to the dataset
# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# preprocessed = [item.strip() for item in preprocessed if item.strip()]
# len(preprocessed) # is going to be 4690 

# all_words = sorted(set(preprocessed))
# vocab_size = len(all_words) # - 1130 

# vocab = {token:integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break

# # simple tokenizer
# class SimpleTokenizerV1:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s,i in vocab.items()}
#     def encode(self, text):
#         preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text) # to numerical values 
#         preprocessed = [
#         item.strip() for item in preprocessed if item.strip()
#         ]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
#     def decode(self, ids): # back to natural language
#         text = " ".join([self.int_to_str[i] for i in ids])
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return text
    
# tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know,"
# Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# # print(ids) # 1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]

# # adding special context tokens :- that is we add <|unk|> to new or unknown words while <|endoftext|> for sentence completionb

# # modifying the vocabulary 
# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# vocab = {token:integer for integer,token in enumerate(all_tokens)}
# # print(len(vocab.items())) - now it's 1132

# # tokenizer with special context words
# class SimpleTokenizerV2:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = { i:s for s,i in vocab.items()}
#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#         preprocessed = [
#         item.strip() for item in preprocessed if item.strip()
#         ]
#         preprocessed = [item if item in self.str_to_int
#         else "<|unk|>" for item in preprocessed]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
#         return text
    
# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# # print(text) Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.

# tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text)) # 1131 = <|unk|> and 1130 = <|endoftext|>
# out is going to be [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]

# now replace brute force way of tokenzing the text with tiktoken

from importlib.metadata import version
import tiktoken

tik_tokenizer = tiktoken.get_encoding('gpt2')

text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
)
# integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers) # [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 27271, 13]

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tik_tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=num_workers
    )
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch) # [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]

second_batch = next(data_iter)  

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False
    )
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer) #  (6,3)

# usefull embdng sizes and encode the input tokens into a
# 256-dimensional vector representation, which is smaller than what the original GPT-3
# model used (in GPT-3, the embedding size is 12,288 dimensions) but still reasonable
# for experimentation. furthermore, we assume that the token IDs were created by the
# BPE tokenizer we implemented earlier, which has a vocabulary size of 50,257:
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
raw_text, batch_size=8, max_length=max_length,
stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape) # ([8, 4])

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape) # ([8, 4, 256])

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape) # ([4, 256])

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape) # ([8, 4, 256])