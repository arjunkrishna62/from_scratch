import torch

inputs = torch.tensor(
[[0.43, 0.15, 0.89], # your     (x^1)
 [0.55, 0.87, 0.66], # journey  (x^2)
 [0.55, 0.87, 0.66], # starts   (x^3)
 [0.22, 0.58, 0.33], # with     (x^4)
 [0.77, 0.25, 0.10], # one      (x^5)
 [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1] 
d_in = inputs.shape[1] # input embedding size , d = 3
d_out = 2 # output embedding size

# torch.manual_seed(123)
# W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# query_2 = x_2 @ W_query
# key_2 = x_2 @ W_key
# value_2 = x_2 @ W_value
# print(query_2)

# keys = inputs @ W_key
# values = inputs @ W_value
# print("keys.shape:", keys.shape)
# print("values.shape:", values.shape)

# keys_2 = keys[1]
# attn_score_22 = query_2.dot(keys_2)
# print(attn_score_22) # unnormalized attention score

# attn_scores_2 = query_2 @ keys.T # computaton to all attention scores
# print(attn_scores_2) # 2nd element matches we computed prev (attn_scores_22)

# d_k = keys.shape[-1] 
# attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) # qk^T / root_dk(dim of the key matrix)
# print(attn_weights_2)

# context_vec_2 = attn_weights_2 @ values
# print('context_vector - > ',context_vec_2)

# self attention class

import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        values = x @ self.W_value
        queries = x @ self.W_query

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print('using torch.nn.Parameter - >  ',sa_v1(inputs))

# self-attention using linear layer.
# instead of manually implementing nn.Parameter(torch.rand(...)) is that nn.Linear
# has an optimized weight initialization scheme, contributing to more stable and
# effective model training.

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax( attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
print('using torch.Linear - >   ',sa_v2(inputs)) # SelfAttention_v1 and SelfAttention_v2 give different outputs because
                                                # they use different initial weights for the weight matrices since nn.Linear uses a more
                                                # sophisticated weight initialization scheme.

