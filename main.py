import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchinfo import summary

# Import functions/classes from your modules
from data_utils import build_dataloaders, en_tokenizer, fr_tokenizer
from transformer import TranslationTransformer
from decoder_only import DecoderOnlyTranslationTransformer # If testing decoder-only
from training import train_model

# --- Configuration ---
# Use a dictionary to hold all hyperparameters
config = {
    # Data parameters
    'max_sequence_length': 50, # Increased sequence length
    'min_token_freq': 5,       # Lower frequency threshold
    'batch_size': 64,         # Adjusted batch size
    'data_url': "http://www.manythings.org/anki/fra-eng.zip",
    'data_dir': ".",

    # Model parameters
    'model_type': 'encoder_decoder', # 'encoder_decoder' or 'decoder_only'
    'n_heads': 8,
    'dim_embedding': 512,     # Standard Transformer embedding size
    'dim_hidden': 2048,       # Standard Transformer feedforward size
    'n_layers': 6,            # Standard Transformer layers
    'dropout': 0.1,

    # Training parameters
    'epochs': 20,
    'lr': 1e-4, # Often start lower for Transformers
    'betas': (0.9, 0.98), # Common Adam betas for Transformers
    'clip': 1.0, # Gradient clipping is common
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'log_every': 100,  # Log metrics every 100 batches

    # WandB parameters (optional)
    'wandb_project': 'INF8225-Translation',
    'wandb_group': 'Transformer-Baseline', # Example group name
    'wandb_run_name': None # Let WandB generate a name or set one
}

# Set seed for reproducibility
torch.manual_seed(config['seed'])
if config['device'] == 'cuda':
    torch.cuda.manual_seed(config['seed'])

# --- Data Loading ---
print("Building dataloaders...")
train_loader, val_loader, en_vocab, fr_vocab, src_pad_idx, tgt_pad_idx = build_dataloaders(
    max_sequence_length=config['max_sequence_length'],
    min_token_freq=config['min_token_freq'],
    batch_size=config['batch_size'],
    data_url=config['data_url'],
    data_dir=config['data_dir']
)

# Add vocab/padding info to config (needed by training loop and model)
config['n_tokens_src'] = len(en_vocab)
config['n_tokens_tgt'] = len(fr_vocab)
config['src_pad_idx'] = src_pad_idx
config['tgt_pad_idx'] = tgt_pad_idx
config['src_vocab'] = en_vocab # Needed for inference logging
config['tgt_vocab'] = fr_vocab # Needed for inference logging
config['src_tokenizer'] = en_tokenizer # Needed for inference logging

print(f"Device: {config['device']}")
print(f"Source Vocab Size: {config['n_tokens_src']}")
print(f"Target Vocab Size: {config['n_tokens_tgt']}")

# --- Model Initialization ---
print(f"Initializing model ({config['model_type']})...")
if config['model_type'] == 'encoder_decoder':
    model = TranslationTransformer(
        n_tokens_src=config['n_tokens_src'],
        n_tokens_tgt=config['n_tokens_tgt'],
        n_heads=config['n_heads'],
        dim_embedding=config['dim_embedding'],
        dim_hidden=config['dim_hidden'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        src_pad_idx=config['src_pad_idx'],
        tgt_pad_idx=config['tgt_pad_idx'],
        device=config['device']
    )
    # Input shape for summary needs target sequence length too
    summary_input_size = [
            (config['batch_size'], config['max_sequence_length']), # Source
            (config['batch_size'], config['max_sequence_length'])  # Target
        ]
    summary_dtypes = [torch.long, torch.long]

elif config['model_type'] == 'decoder_only':
    # Note: Decoder-only might use a combined vocab or handle source differently
    # Assuming it takes target sequence for now
    model = DecoderOnlyTranslationTransformer(
        n_tokens_vocab=config['n_tokens_tgt'], # Using target vocab size
        n_heads=config['n_heads'],
        dim_embedding=config['dim_embedding'],
        dim_hidden=config['dim_hidden'],
        num_layers=config['n_layers'],
        dropout=config['dropout'],
        pad_idx=config['tgt_pad_idx'],
        device=config['device']
    )
    summary_input_size = [(config['batch_size'], config['max_sequence_length'])] # Just one input sequence
    summary_dtypes=[torch.long]
else:
     raise ValueError("Invalid model_type in config.")

model.to(config['device'])

# --- Optimizer and Loss ---
config['optimizer'] = optim.Adam(
    model.parameters(),
    lr=config['lr'],
    betas=config['betas'],
)

# Optional: Learning rate scheduler (common for Transformers)
# scheduler = optim.lr_scheduler.StepLR(config['optimizer'], step_size=1, gamma=0.95) # Example

# Use ignore_index for padding in CrossEntropyLoss
config['loss'] = nn.CrossEntropyLoss(ignore_index=config['tgt_pad_idx'])


# --- Model Summary ---
print("Model Summary:")
try:
    summary(
        model,
        input_size=summary_input_size,
        dtypes=summary_dtypes,
        device=config['device'],
        depth=3,
    )
except Exception as e:
    print(f"Could not generate model summary: {e}")


# --- Training ---
# Initialize WandB (optional)
if config.get('wandb_project'):
     wandb.init(
         project=config['wandb_project'],
         config=config, # Log hyperparameters
         group=config.get('wandb_group'),
         name=config.get('wandb_run_name'),
         save_code=True,
     )
     # Optional: Watch model gradients
     # wandb.watch(model, log_freq=100)

# Start training loop
train_model(model, config)

# Finish WandB run
if wandb.run:
    wandb.finish()

print("Script finished.")
