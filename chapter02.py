#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)


# In[2]:


with open("the-verdict.txt", "r", encoding="utf-8") as f:raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])


# In[3]:


import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)


# In[4]:


result = re.split(r'([,.]|\s)', text)
print(result)


# In[5]:


result = [item for item in result if item.strip()]
result


# In[6]:


text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
print(result)
result = [item.strip() for item in result if item.strip()]
print(result)


# In[7]:


# applying it to the dataset
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
len(preprocessed)


# In[8]:


preprocessed[:10]


# In[9]:


all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
vocab_size


# In[10]:


vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


# In[11]:


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text) # to numerical values 
        preprocessed = [
        item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids): # back to natural language
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# In[12]:


tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)


# In[13]:


print(tokenizer.decode(ids))


# In[14]:


text = "Hello, do you like tea?"
print(tokenizer.encode(text)) # error due to the text is not in training set. The word 'hello' not used in the short story 'The verdict'


# In[15]:


# adding special context tokens :- that is we add <|unk|> to new or unknown words while <|endoftext|> for sentence completionb


# In[16]:


# modifying the vocabulary 
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))


# In[17]:


for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


# In[18]:


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
        item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


# In[19]:


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)


# In[20]:


tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text)) # 1131 = <|unk|> and 1130 = <|endoftext|>


# In[21]:


print(tokenizer.decode(tokenizer.encode(text)))


# In[22]:


from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))


# In[23]:


tik_tokenizer = tiktoken.get_encoding('gpt2')


# In[24]:


text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."
)
integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)


# In[25]:


string = tokenizer.decode(integers)
string


# In[26]:


unk = "Akwirw ier."
print(tik_tokenizer.encode(unk))


# In[27]:


print(tokenizer.decode(tik_tokenizer.encode(unk)))


# In[28]:


# now tokenize the short story dataset using the BPE tokenizer
enc_text = tik_tokenizer.encode(raw_text)
print(len(enc_text))


# In[29]:


enc_sample = enc_text[50:]
len(enc_sample)


# In[30]:


context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y: {y}")


# In[31]:


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)


# In[32]:


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


# In[33]:


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


# In[ ]:





# In[34]:


def create_dataloader_v1(txt, batch_size=4, max_length=256,
stride=128, shuffle=True, drop_last=True,
num_workers=0):
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


# In[35]:


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)


# In[36]:


second_batch = next(data_iter)
second_batch


# In[37]:


dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False
    )
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


# In[38]:


# token IDs into token embedding vectors

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)


# In[39]:


print(embedding_layer(torch.tensor([3]))) # embedding vector


# In[40]:


print(embedding_layer(input_ids))


# In[41]:


# usefull embdng sizes and encode the input tokens into a
# 256-dimensional vector representation, which is smaller than what the original GPT-3
# model used (in GPT-3, the embedding size is 12,288 dimensions) but still reasonable
# for experimentation. furthermore, we assume that the token IDs were created by the
# BPE tokenizer we implemented earlier, which has a vocabulary size of 50,257:
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


# In[42]:


max_length = 4
dataloader = create_dataloader_v1(
raw_text, batch_size=8, max_length=max_length,
stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)


# In[43]:


token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)


# In[44]:


context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)


# In[45]:


input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)


# In[46]:


# self-attention intro


# In[47]:


import torch
inputs = torch.tensor(
[[0.43, 0.15, 0.89], # your     (x^1)
 [0.55, 0.87, 0.66], # journey  (x^2)
 [0.55, 0.87, 0.66], # starts   (x^3)
 [0.22, 0.58, 0.33], # with     (x^4)
 [0.77, 0.25, 0.10], # one      (x^5)
 [0.05, 0.80, 0.55]] # step     (x^6)
)


# In[48]:


inputs.shape[1]


# In[49]:


x_2 = inputs[1] 
d_in = inputs.shape[1] # input embedding size , d = 3
d_out = 2 # output embedding size


# In[50]:


torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)


# In[51]:


W_query


# In[52]:


query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)


# In[53]:


keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)


# In[54]:


keys


# In[55]:


keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22) # unnormalized attention score


# In[56]:


attn_scores_2 = query_2 @ keys.T # computaton to all attention scores
print(attn_scores_2) # 2nd element matches we computed prev (attn_scores_22)


# In[57]:


d_k = keys.shape[-1] 
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) # qk^T / root_dk(dim of the key matrix)
print(attn_weights_2)


# In[58]:


context_vec_2 = attn_weights_2 @ values
print(context_vec_2)


# In[59]:


# compact self-attention class


# In[60]:


import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax( attn_scores / keys.shape[-1]**0.5, dim=-1 )
        context_vec = attn_weights @ values
        return context_vec


# In[61]:


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs)) # inputs contains 6 embedding vectors , results in a matrix storing 6 context vectors


# In[62]:


# self-attention using linear layer.
# instead of manually implementing nn.Parameter(torch.rand(...)) is that nn.Linear
# has an optimized weight initialization scheme, contributing to more stable and
# effective model training.


# In[63]:


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax( attn_scores / keys.shape[-1]**0.5, dim=-1 )
        context_vec = attn_weights @ values
        return context_vec


# In[64]:


torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs)) # SelfAttention_v1 and SelfAttention_v2 give different outputs because
                     # they use different initial weights for the weight matrices since nn.Linear uses a more
                     # sophisticated weight initialization scheme.


# In[65]:


queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
                                # Reuses the query and key weight matrices
                                # of the SelfAttention_v2 object from the
                                # previous section for convenience
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)


# In[66]:


context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length)) #tril fn to create a mask where the values above the diagonal are zero
print(mask_simple)


# In[67]:


masked_simple = attn_weights * mask_simple
masked_simple


# In[68]:


row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm) # renormalize the attention weights to sum up to 1 again in each row


# In[69]:


row_sums


# In[70]:


mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)


# In[71]:


attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)


# In[72]:


torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))


# In[73]:


torch.manual_seed(123)
print(dropout(attn_weights))


# In[74]:


batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)


# In[75]:


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer( # buffers are used automatically move to the apropriate device (cpu or gpu) along with the model
        'mask',
        torch.triu(torch.ones(context_length, context_length),
        diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


# In[76]:


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)


# In[77]:


context_vecs


# In[78]:


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
        dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1) # processing sequentially not simultaneously


# In[79]:


torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2


# In[80]:


mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# In[81]:


# tensor is two because we have 2 input texts  (the input texts are duplicated, which is why the context vectors are exactly the same for those)
# second dim refers to the 6 tokens in each input
# third dim refers to the 4 dim embedding of each token


# In[82]:


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
        context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # reduces the projection dim to match the desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # uses a Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                diagonal=1)
        )
        
    # tensor shape: (b, num_tokens, d_out)
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # we implicitly split the matrix by adding a num_heads dimension.
        #then we unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim).
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )
        #transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2) 
        values = values.transpose(1, 2)
    
        attn_scores = queries @ keys.transpose(2, 3) # computes dot for each head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # masks truncated to the no of tokens
    
        attn_scores.masked_fill_(mask_bool, -torch.inf) # uses mask to fill attn_scores
    
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
    
        context_vec = (attn_weights @ values).transpose(1, 2) # tensor shape: (b, num_tokens, n_heads, head_dim)
    
        # Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
        b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec) # adds an optional linear projection
        return context_vec


# In[83]:


a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
[0.8993, 0.0390, 0.9268, 0.7388],
[0.7179, 0.7058, 0.9156, 0.4340]],
[[0.0772, 0.3565, 0.1479, 0.5331],
[0.4066, 0.2318, 0.4545, 0.9737],
[0.4606, 0.5159, 0.4220, 0.5786]]]])


# In[ ]:





# In[84]:


print(a @ a.transpose(2, 3))


# In[85]:


first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_res)
second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)


# In[86]:


a.shape


# In[87]:


torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# In[90]:


import torch
import torch.nn as nn

GPT_CONFIG_124M = {
"vocab_size": 50257,     # Vocabulary size
"context_length": 1024,  # Context length
"emb_dim": 768,          # Embedding dimension
"n_heads": 12,           # Number of attention heads
"n_layers": 12,          # Number of layers
"drop_rate": 0.1,        # Dropout rate
"qkv_bias": False        # Query-Key-Value bias
}


# In[91]:


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
            for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
        torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# In[92]:


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x):
        return x
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x


# In[93]:


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)


# In[94]:


torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch) # logits are unormalized model's pred before an actvn fn is applied , these values can range from -ve to +ve infinity
print("Output shape:", logits.shape) # and represent the model's confidence in assigning an input to a particular class.
logits 


# In[95]:


torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
out


# In[96]:


batch_example


# In[97]:


mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)


# In[98]:


out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)


# In[99]:


torch.set_printoptions(sci_mode=False) 
print("Mean:\n", mean)
print("Variance:\n", var)


# In[100]:


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# In[101]:


ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)


# In[102]:


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# In[103]:


import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_lay
plt.show


# In[ ]:


# The smoothness of GELU can lead to better optimization properties during training,
# as it allows for more nuanced adjustments to the model’s parameters. In contrast,
# ReLU has a sharp corner at zero , which can sometimes make opti-mization harder, 
# especially in networks that are very deep or have complex architectures.
# Moreover, unlike ReLU, which outputs zero for any negative input, GELU
# allows for a small, non-zero output for negative values. This characteristic means that
# during the training process, neurons that receive negative input can still contribute to
# the learning process, albeit to a lesser extent than positive inputs.


# In[104]:


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)


# In[105]:


ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)


# In[106]:


# Shortcut Connections or Skip Conncetions or Residual Connections
# as the layer progress there's a high chance of problem like vanishing gradient.
# vanshing gradient :- vanishing gradient problem refers to the issue where gradients
#     (which guide weight updates during training) become progressively smaller as they
#     propagate backward through the layers, making it difficult to effectively train earlier
#     layers.
# to prevent vanishing gradiant prblem  the soln is  skip or residual conncetions :-
#         creates alternative or shotcut path for grdient flow  thruogh the network by skipping one or more layers,
#         which is achieved by adding output of one layer to the output of a later layer


# In[107]:


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
        nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
        GELU())
        ])
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x) # compute the output of the current layer
            if self.use_shortcut and x.shape == layer_output.shape: # check if shortcuts can be applied
                x = x + layer_output 
            else:
                x = layer_output
        return x


# In[108]:


layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123) # specifies random seeds for initial weights for reproducibility
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False

)


# In[109]:


def print_gradients(model, x):
    output = model(x) # fwd pass
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target) # calculates loss based on how close the target and output are
    loss.backward() # backward pass to calculate the gradients
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


# In[ ]:





# In[ ]:





# In[110]:


print_gradients(model_without_shortcut, sample_input)


# In[111]:


# the gradient becomes smaller when we progress from the last layer (vanishing gradient)


# In[112]:


torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
layer_sizes, use_shortcut=True # applying skip connections
)
print_gradients(model_with_shortcut, sample_input)


# In[113]:


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        
        shortcut = x # shortcut conn for attention block
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # add the orginal input back

        shortcut = x # shortcut conn for ff block
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # adds the orginal back
        return x
                


# In[114]:


torch.manual_seed(123)
x = torch.rand(2, 4, 768) # smaple i/p shape [batch_size, num_tokens, num_emb]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)


# In[115]:


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb( 
        torch.arange(seq_len, device=in_idx.device) # device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits on.
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# In[116]:


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)


# In[117]:


total_params = sum(p.numel() for p in model.parameters()) # analyzing size using numel() - number of elements, we can collect
print(f"Total number of parameters: {total_params:,}") # total number of parameters in the model's parameter tensors.


# In[118]:


# why 163m instead of 124m ? - reason is a concept called "weight tying" which was used in the original GPT-2 architecture.
# It means that the original GPT-2 architecture reuses the weights from the tokn embdng layer in its output layer.
# To understand better, let’s take a look at the shapes of the token emdng layer and linear ouput layer that we initialzed
# on the model via the GPTModel earlier:


# In[119]:


print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)


# In[120]:


# token embedding and output layers are very large due to the number of rows for
# the 50,257 in the tokenizer’s vocabulary. Let’s remove the output layer parameter
# count from the total GPT-2 model count according to the weight tying:


# In[121]:


total_params_gpt2 = (
total_params - sum(p.numel()
for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
f"considering weight tying: {total_params_gpt2:,}"
)


# In[122]:


# number of parameters in feed forward and attention modules:


# In[123]:


ffn_param = sum(p.numel() for p in ffn.parameters())
print(f"feed fwd parameteres: {ffn_param:,}")
ffn


# In[124]:


mha_param = sum(p.numel() for p in mha.parameters())
print(f"MultiHeadAttn parameters: {mha_param:,}")
mha


# In[125]:


(768 * 3072) * 2 # ffnn


# In[126]:


((3 * 768) * 3) + (768 * 768) # mha


# In[127]:


# lastly compute the memory req for 163m
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")


# In[128]:


# initializing larger gpt models, first gpt_medium with 24 tfr blocks


# In[129]:


GPT_CONFIG_MEDIUM = {
"vocab_size": 50257,     # Vocabulary size
"context_length": 1024,  # Context length
"emb_dim": 1024,          # Embedding dimension
"n_heads": 16,           # Number of attention heads
"n_layers": 24,          # Number of layers
"drop_rate": 0.1,        # Dropout rate
"qkv_bias": False        # Query-Key-Value bias
}


# In[130]:


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb( 
        torch.arange(seq_len, device=in_idx.device) # device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits on.
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# In[131]:


torch.manual_seed(123)
gpt_medium = GPTModel(GPT_CONFIG_MEDIUM)
medium_out = gpt_medium(batch)
print(medium_out.shape)
medium_out


# In[132]:


ttl_gpt_medium_params = sum(p.numel() for p in gpt_medium.parameters()) # small model has 163,009,536 , 24 block has 406m
print(f"{ttl_gpt_medium_params:,}")


# In[133]:


ttl_gpt_medium_params = (
total_params - sum(p.numel()
for p in gpt_medium.out_head.parameters())
)
print(f"Number of trainable parameters "
f"considering weight tying: {ttl_gpt_medium_params:,}"
)


# In[134]:


# gpt_large  with 36 tfr blocks


# In[135]:


GPT_CONFIG_LARGE = {
"vocab_size": 50257,     # Vocabulary size
"context_length": 1024,  # Context length
"emb_dim": 1280,          # Embedding dimension
"n_heads": 20,           # Number of attention heads
"n_layers": 36,          # Number of layers
"drop_rate": 0.1,        # Dropout rate
"qkv_bias": False        # Query-Key-Value bias
}

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb( 
        torch.arange(seq_len, device=in_idx.device) # device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits on.
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# In[136]:


torch.manual_seed(123)
gpt_large = GPTModel(GPT_CONFIG_LARGE)
large_out = gpt_large(batch)
print(large_out.shape)
large_out


# In[137]:


ttl_gpt_large_params = sum(p.numel() for p in gpt_large.parameters()) 
print(f"{ttl_gpt_large_params:,}")


# In[138]:


ttl_gpt_large_params = (
total_params - sum(p.numel()
for p in gpt_large.out_head.parameters())
)
print(f"Number of trainable parameters "
f"considering weight tying: {ttl_gpt_large_params:,}"
)


# In[139]:


# gpt xl with tfr blocks 32


# In[140]:


GPT_CONFIG_XL = {
"vocab_size": 50257,     # Vocabulary size
"context_length": 1024,  # Context length
"emb_dim": 1600,          # Embedding dimension
"n_heads": 25,           # Number of attention heads
"n_layers": 48,          # Number of layers
"drop_rate": 0.1,        # Dropout rate
"qkv_bias": False        # Query-Key-Value bias
}

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb( 
        torch.arange(seq_len, device=in_idx.device) # device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits on.
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# In[141]:


torch.manual_seed(123)
gpt_xl = GPTModel(GPT_CONFIG_XL)
xl_out = gpt_xl(batch)
print(xl_out.shape)
xl_out


# In[151]:


ttl_gpt_large_params = sum(p.numel() for p in gpt_xl.parameters()) 
print(f"{ttl_gpt_large_params:,}") 


# In[152]:


ttl_gpt_xl_params = (
total_params - sum(p.numel()
for p in gpt_xl.out_head.parameters())
)
print(f"Number of trainable parameters "
f"considering weight tying: {ttl_gpt_xl_params:,}"
)


# In[154]:


def generate_text_simple(model, idx, # dx is a (batch, n_tokens) array of indices in the current context.
    max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # crops current context if exceeds the supported context size, if llm supports only 5 tokens, and the context is 10, then only the last 5 tokens are used as context
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] # Focuses only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1) # prob has shape (batch, vocab_size).
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # idx next has shape(batch, 1)
        idx = torch.cat((idx, idx_next), dim=1) # appends sampled text to the running sequence, where idx has shape(batch, n_tokens+1)
    return idx


# In[155]:


start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) # adds batch dim
print("encoded_tensor.shape:", encoded_tensor.shape)


# In[156]:


model.eval() # Disables dropout since we are not training the model
out = generate_text_simple(
model=model,
idx=encoded_tensor,
max_new_tokens=6,
context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))


# In[157]:


decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)


# In[ ]:




