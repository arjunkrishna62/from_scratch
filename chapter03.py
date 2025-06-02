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
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# print('nn.parameter\n',sa_v1(inputs))

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


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print('using torch.Linear - >   ',sa_v2(inputs)) # SelfAttention_v1 and SelfAttention_v2 give different outputs because
                                                # they use different initial weights for the weight matrices since nn.Linear uses a more
                                                # sophisticated weight initialization scheme.

# queries = sa_v2.W_query(inputs)
# keys = sa_v2.W_key(inputs)
#                                     # Reuses the query and key weight matrices  of the SelfAttention_v2 object from the  previous section for convenienc 
# attn_scores = queries @ keys.T
# attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# print(attn_weights)

# # using tril() to mask the elements above the diagonal
# context_length = attn_scores.shape[0]
# mask_simple = torch.tril(torch.ones(context_length, context_length))
# print(mask_simple)

# # now masked attention weights
# masked_simple = attn_weights * mask_simple
# print(masked_simple)

# # renormalize the attention weights to sum up to 1 again in each row
# row_sums = masked_simple.sum(dim=-1, keepdim=True)
# masked_simple_norm = masked_simple / row_sums
# print(masked_simple_norm)

# mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
# masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# print(masked)

# attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
# print(attn_weights)

# # dropout

# torch.manual_seed(123)
# dropout = torch.nn.Dropout(0.5)
# example = torch.ones(6, 6)
# print('dropout \n',dropout(example))

# torch.manual_seed(123)
# print(dropout(attn_weights))

# implementing a compact causal attention class

batch = torch.stack((inputs, inputs), dim=0)
print('batch shape -> ',batch.shape)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
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
    
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
# print("context_vecs.shape:", context_vecs.shape)

# mha wrapper
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
# context_vecs = mha(batch)
# print('context_vector \n',context_vecs)
# print("context_vecs.shape:", context_vecs.shape)

# multihead attention with weight splits

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # reduces the projection dim to match the desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # uses a linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # masks truncated to number of tokens

        attn_scores.masked_fill_(mask_bool, -torch.inf) # uses masks to fill attention scores

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) # tensor shape: (b, num_tokens, n_heads, head_dim)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec) # adds an optional linear projection
        return context_vec

# to illustrate batches matrix multiplicaton eg:
# a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
#                     [0.8993, 0.0390, 0.9268, 0.7388],
#                     [0.7179, 0.7058, 0.9156, 0.4340]],

#                     [[0.0772, 0.3565, 0.1479, 0.5331],
#                     [0.4066, 0.2318, 0.4545, 0.9737],
#                     [0.4606, 0.5159, 0.4220, 0.5786]]]])

# first_head = a[0, 0, :, :]
# first_res = first_head @ first_head.T
# print("First head:\n", first_res)

# second_head = a[0, 1, :, :]
# second_res = second_head @ second_head.T
# print("\nSecond head:\n", second_res)

# print('same as above : ',a @ a.transpose(2, 3)) # will obtain same as the above batched mat_mul 

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

