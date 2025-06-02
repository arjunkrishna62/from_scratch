import torch
from chapter04 import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, # 1024 to 256
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1, # It’s possible and common to set dropout to 0.
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

# Utility functions for text to token ID conversion

import tiktoken
from chapter04 import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # .unsqueeze(0) adds the batch dimension
    return encoded_tensor
    
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # removes batch dimension
    return tokenizer.decode(flat.tolist())
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# calculating text generation loss
inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
[40, 1107, 588]]) # "I really like"

targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
[1107, 588, 11311]]) # " really like chocolate"]

with torch.no_grad(): # not training so dsable gradient tracking
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1) # prob of each token in vocabulary
print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# print("Token IDs:\n", token_ids)

# print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
# print(f"Outputs batch 1: "f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# print the initial softmax probability scores corresponding to the target tokens 
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 1:", target_probas_1)
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 2:", target_probas_2)
# calculating the loss for the probability scores of the two example batches by applying logarithms to porbability scores.
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

avg_log_probas = torch.mean(log_probas) # computing the avg log probabilities
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1 # nothing but avg log probability multiplied by -1
print(neg_avg_log_probas)

# cross_entropy loss function in PyTorch, we want to flatten these tensors by combining them over the batch dimension:
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
# print("Flattened logits:", logits_flat.shape)
# print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat) # instead of manually computing the neg avg log problties
# print(loss)
# # Calculating the training and validation set losses from the dataset the verdict text
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
# print("Characters:", total_characters)
# print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

from chapter02 import create_dataloader_v1

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)
# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)

# implement a utility function to calculate the cross entropy loss of a given batch returned via the training and validation loader:
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device) # the transfer to a given device allows us to transfer the data to a GPU.
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

# Function to compute the training and validation loss , calculates loss over all batches
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader) # iteratives over all batches if no fixed num_batches is specified
    else:
        num_batches = min(num_batches, len(data_loader)) # reduces the number of batches to match the total number of batches in the data loader if num_batches exceeds the number of batches in the data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
            input_batch, target_batch, model, device
            )
            total_loss += loss.item() # sum loss for each batch
        else:
            break
    return total_loss / num_batches # avg loss over all batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad(): # disables gradient tracking for efficiency because we are not training yet
    train_loss = calc_loss_loader(train_loader, model, device) # sure the data is loaded onto the same device as the LLM model.
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# Training an LLM

# The main function for pretraining LLMs

def train_model_simple(model, train_loader, val_loader,
    optimizer, device, num_epochs,
    eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], [] # initialized list to track losses and tokens each
    tokens_seen, global_step = 0, -1
    
    for epoch in range(num_epochs): # starts the training loop 
        model.train() # Starts the main training loop
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # resets loss gradients from prev batch iteration
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward() # calc loss gradients
            optimizer.step() # updates model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1
            
            if global_step % eval_freq == 0: # optional evaluation step
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
                generate_and_print_sample( # prints a sample text after each epoch
                    model, tokenizer, device, start_context
                )
    return train_losses, val_losses, track_tokens_seen

# The evaluate_model function prints the training and validation set losses after each model update so we can evaluate whether
# the training improves the model, More specifically, the evaluate_model function calculates the loss over the training and validation set
# while ensuring the model is in evaluation mode with gradient tracking and dropout disabled when calculating the loss
# over the training and validation sets:
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # dropout disabled during eval for stable, reproducible results
    with torch.no_grad(): # disabled which is not required during evaluation, to reduce the computational overhead
        train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

# Similar to evaluate_model, the generate_and_print_sample function is a convenience function that we use to track whether the model improves during the training.
# In particular,the generate_and_print_sample () takes a text snippet (start_context) as Input converts it into token IDs, and feeds it to the LLM to generate a text sample
# using the generate_text_simple function we used earlier:
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# model.to(device)
# optimizer = torch.optim.AdamW(
# model.parameters(), # method returns all trainable weight params of the model
#     lr=0.0004, weight_decay=0.1
# )
# num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=5, eval_iter=5,
#     start_context="Every effort moves you", tokenizer=tokenizer
# )

import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
#     fig, ax1 = plt.subplots(figsize=(5, 3))
#     ax1.plot(epochs_seen, train_losses, label="Training loss")
#     ax1.plot(
#     epochs_seen, val_losses, linestyle="-.", label="Validation loss"
#     )
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend(loc="upper right")
#     ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
#     ax2 = ax1.twiny()
#     ax2.plot(tokens_seen, train_losses, alpha=0)
#     ax2.set_xlabel("Tokens seen")
#     fig.tight_layout()
#     plt.show()
# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
model=model,
idx=text_to_token_ids("Every effort moves you", tokenizer),
max_new_tokens=25,
context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# temperature scaling example :- probabilistic sampling
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}

# assume the LLM is given the start context "every effort moves you" and generates the following next-token logits:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

# we convert the logits into probabilities via the softmax function and obtain the token ID corresponding to the
# generated token via the argmax function, which we can then map back into text via the inverse vocabulary:
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])

# To implement a probabilistic sampling process, we can now replace argmax with the multinomial function in PyTorch:
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])

def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")
print_sampled_tokens(probas)

def softmax_with_temperature(logits, temperature): # temperature scaling is just dividing the logits with number greater than zero
    scaled_logits = logits / temperature # to control the distribution
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5] # orginal, lower and higher confidence
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i],
                bar_width, label=f'Temperature = {T}')
    
# ax.set_ylabel('Probability')
# ax.set_xticks(x)
# ax.set_xticklabels(vocab.keys(), rotation=90)
# ax.legend()
# plt.tight_layout()
# plt.show()

# why top-k? using top-k , telling the model to select only from the top-k samples(highest probability).
# rather than looking for the entire logits, to do this- select highest probabilities samples from the logits
# and the rest will be masked (just like chapter-3 causal attention) 
# replaces with -inf before softmax after the rest of the logits would be assingned 0.00 nd the remaining probabilities sum up to 1

top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1], # identifies logits less than the min in the top 3
    input=torch.tensor(float('-inf')), # assigns -inf to lower logits
    other=next_token_logits # retains the orginal logits for all othre tokens
)
print('logits ',new_logits)

topk_probas = torch.softmax(new_logits, dim=0)
print('top-k probabilities',topk_probas)

# We can now apply the temperature scaling and multinomial function for probabilistic sampling to select the next token among these three non-zero probability scores to
# generate the next token. We do this next by modifying the text generation function.

# Modifying the text generation function
def generate(model, idx, max_new_tokens, context_size,
                temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
                logits = model(idx_cond)
        logits = logits[:, -1, :]
            
        if top_k is not None: # filters logits with top-k sampling 
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0: # applies temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:     # Carries out greedy next-token selection as before when temperature scaling is disabled
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id: # stops generating early if end-of-seq token is encountered
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx