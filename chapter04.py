GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

# The order in which we code the GPT architecture. We start with the GPT
# backbone, a placeholder architecture, before getting to the individual core pieces and
# eventually assembling them in a transformer block for the final GPT architecture.

# A placeholder GPT model architecture class

import torch
import torch.nn as nn

# class DummyGPTModel(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
#         self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
#         self.drop_emb = nn.Dropout(cfg["drop_rate"])
#         self.trf_blocks = nn.Sequential( # placeholder for tfr blocks
#             *[DummyTransformerBlock(cfg)
#             for _ in range(cfg["n_layers"])]
#         )
#         self.final_norm = DummyLayerNorm(cfg["emb_dim"]) # layer norm placeholder 
#         self.out_head = nn.Linear(
#             cfg["emb_dim"], cfg["vocab_size"], bias=False
#         )

#     def forward(self, in_idx):
#         batch_size, seq_len = in_idx.shape
#         tok_embeds = self.tok_emb(in_idx)
#         pos_embeds = self.pos_emb(
#             torch.arange(seq_len, device=in_idx.device)
#         )
#         x = tok_embeds + pos_embeds
#         x = self.drop_emb(x)
#         x = self.trf_blocks(x)
#         x = self.final_norm(x)
#         logits = self.out_head(x)
#         return logits
    
# class DummyTransformerBlock(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#     def forward(self, x):
#         return x
    
# class DummyLayerNorm(nn.Module):
#     def __init__(self, normalized_shape, eps=1e-5):
#         super().__init__()
#     def forward(self, x):
#         return x
    

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# torch.manual_seed(123)
# model = DummyGPTModel(GPT_CONFIG_124M)
# logits = model(batch)
# print("Output shape:", logits.shape)
# print('from 124m :- ',logits)

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

# before layer norm compute mean and var

# mean = out.mean(dim=-1, keepdim=True)
# var = out.var(dim=-1, keepdim=True)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# # applying layer noramlization to the output

# out_norm = (out - mean) / torch.sqrt(var)
# mean = out_norm.mean(dim=-1, keepdim=True)
# var = out_norm.var(dim=-1, keepdim=True)
# print("Normalized layer outputs:\n", out_norm)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# note that the value –5.9605e-08 in the output tensor is the scientific notation for
# –5.9605 × 10-8, which is –0.000000059605 in decimal form. This value is very close to 0,
# but it is not exactly 0 due to small numerical errors that can accumulate because of
# the finite precision with which computers represent numbers.

# after removiing the scientific notation
# torch.set_printoptions(sci_mode=False)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# Layer noramalization classs
# This specific implementation of layer normalization operates on the last dimension of the input tensor x, which represents embedding dim (emb_dim).

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

# results show that the layer normalization code works as expected and normalizes
# the values of each of the two inputs such that they have a mean of 0 and a variance of 1:

# ln = LayerNorm(emb_dim=5)
# out_ln = ln(batch_example)
# mean = out_ln.mean(dim=-1, keepdim=True)
# var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# Implementing a feed forward network with GELU activations (instead of traditional ReLu)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
import matplotlib.pyplot as plt
# gelu, relu = GELU(), nn.ReLU()

# x = torch.linspace(-3, 3, 100)
# y_gelu, y_relu = gelu(x), relu(x)
# plt.figure(figsize=(8, 3))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True)
# plt.tight_layout()
# plt.show()

# Feed Forward Neural Network Module

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


ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
# print(out.shape)

# # Shortcut Connections or Skip Conncetions or Residual Connections
# as the layer progress there's a high chance of problem like vanishing gradient.
# vanshing gradient :- vanishing gradient problem refers to the issue where gradients
#     (which guide weight updates during training) become progressively smaller as they
#     propagate backward through the layers, making it difficult to effectively train earlier
#     layers.
# to prevent vanishing gradiant prblem  the soln is  skip or residual conncetions :-
#         creates alternative or shotcut path for grdient flow  thruogh the network by skipping one or more layers,
#         which is achieved by adding output of one layer to the output of a later layer

# class ExampleDeepNeuralNetwork(nn.Module):
#     def __init__(self, layer_sizes, use_shortcut):
#         super().__init__()
#         self.use_shortcut = use_shortcut
#         self.layers = nn.ModuleList([
#         nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
#         GELU()),
#         nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
#         GELU()),
#         nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
#         GELU()),
#         nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
#         GELU()),
#         nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
#         GELU())
#         ])
#     def forward(self, x):
#         for layer in self.layers:
#             layer_output = layer(x) # compute the output of the current layer
#             if self.use_shortcut and x.shape == layer_output.shape: # check if shortcuts can be applied
#                 x = x + layer_output 
#             else:
#                 x = layer_output
#         return x


# layer_sizes = [3, 3, 3, 3, 3, 1]
# sample_input = torch.tensor([[1., 0., -1.]])
# torch.manual_seed(123)
# model_without_shortcut = ExampleDeepNeuralNetwork(
#     layer_sizes, use_shortcut=False
# )

# def print_gradients(model, x):
#     output = model(x) # fwd pass
#     target = torch.tensor([[0.]])

#     loss = nn.MSELoss()
#     loss = loss(output, target) # calculates loss based on how close the target and output are

#     loss.backward() # backward pass to calculate the gradients

#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# print_gradients(model_without_shortcut, sample_input) #  the gradients become smaller as we progress from the last layer (layers.4) to the first layer (layers.0)

# # so model with skip connections
# torch.manual_seed(123)
# model_with_shortcut = ExampleDeepNeuralNetwork(
#     layer_sizes, use_shortcut=True
# )
# print('after applying skip connectiosn..')
# print_gradients(model_with_shortcut, sample_input)

from chapter03 import MultiHeadAttention

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
    
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)

# GPT Model

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
    
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters()) # numel() numbfer of elemtnts 
print(f"Total number of parameters: {total_params:,}") # 163m instead of 124m why?

# concept called weight tying, which was used in the original GPT-2 architecture.
# It means that the original GPT-2 architecture reuses the weights from the token embedding layer in its output layer.

print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

# Token embedding layer shape: torch.Size([50257, 768])
# Output layer shape: torch.Size([50257, 768])

# The token embedding and output layers are very large due to the number of rows for
# the 50,257 in the tokenizer’s vocabulary. Let’s remove the output layer parameter
# count from the total GPT-2 model count according to the weight tying:

total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters())
)
print(f"Number of trainable parameters " f"considering weight tying: {total_params_gpt2:,}" ) # outputts 124m

# generating sample text 

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

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) # adds batch dim
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval() # Disables dropout since we are not training the model
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))

# using the .decode method of the tokenzer, to convert token IDs back into text.

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print('decoded text :- ',decoded_text) # he model generated gibberish, which is not at all like the coherent text