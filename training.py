import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import wandb # Assuming wandb is used for logging
from torchtext.vocab import Vocab
from tqdm import tqdm # Use tqdm for progress bars
# Assuming inference.py with beam_search is available
from inference import beam_search, beautify

def print_logs(dataset_type: str, logs: dict, epoch: int, total_epochs: int):
    """Prints formatted logs."""
    desc = [f'{name}: {value:.4f}' for name, value in logs.items()] # Increased precision
    desc_str = '    '.join(desc)
    print(f'Epoch {epoch+1}/{total_epochs} - {dataset_type} -\t{desc_str}')


def topk_accuracy(
        logits: torch.FloatTensor, # Changed input to logits
        targets: torch.LongTensor, # Changed name
        k: int,
        ignore_index: int, # Use ignore_index directly
    ) -> torch.FloatTensor:
    """Compute the top-k accuracy, ignoring specified index."""
    # logits shape: [N, vocab_size], targets shape: [N]
    # N = batch_size * seq_len

    # Mask for valid (non-ignored) tokens
    valid_mask = (targets != ignore_index)
    num_valid = valid_mask.sum().item()

    if num_valid == 0:
        return torch.tensor(0.0, device=logits.device) # Avoid division by zero

    # Get top k predictions
    _, pred_tokens = logits.topk(k=k, dim=-1) # [N, k]

    # Expand targets to compare with each of the top k predictions
    targets_expanded = targets.unsqueeze(1).expand_as(pred_tokens) # [N, k]

    # Check if the true target is within the top k predictions
    correct = (pred_tokens == targets_expanded)

    # Consider only valid tokens for accuracy calculation
    correct_valid = correct[valid_mask] # Shape: [num_valid, k]

    # A token is correctly predicted if *any* of the top k match
    correct_k = correct_valid.any(dim=1) # Shape: [num_valid]

    # Calculate accuracy
    acc = correct_k.sum().float() / num_valid
    return acc


def loss_batch(
        model: nn.Module,
        source: torch.LongTensor,
        target: torch.LongTensor,
        loss_fn: nn.Module, # Pass loss_fn directly
        tgt_pad_idx: int, # Pass pad index
        device: str,
        model_type: str = 'encoder_decoder' # Add model type flag
    )-> dict:
    """Compute the metrics associated with this batch."""
    model.to(device)
    source, target = source.to(device), target.to(device)

    # Prepare target input (exclude <eos>) and output (exclude <bos>)
    target_in = target[:, :-1]
    target_out = target[:, 1:] # Ground truth for loss calculation

    # Forward pass
    if model_type == 'encoder_decoder':
        logits = model(source, target_in) # [batch_size, n_tgt_tokens-1, n_vocab]
    elif model_type == 'decoder_only':
         # Assuming decoder-only takes the combined sequence or just target
         # If just target, need shifting like above. If combined, need careful slicing.
         # Let's assume it takes target_in and predicts next tokens for simplicity here.
         # Adjust this logic based on how DecoderOnlyTranslationTransformer is implemented.
         logits = model(target_in) # [batch_size, n_tgt_tokens-1, n_vocab]
    else:
         raise ValueError("Invalid model_type")


    # Reshape for loss calculation
    # Logits: [batch_size * (n_tgt_tokens - 1), n_vocab]
    # Target_out: [batch_size * (n_tgt_tokens - 1)]
    logits_flat = logits.reshape(-1, logits.shape[-1])
    target_out_flat = target_out.reshape(-1)

    # Calculate loss (CrossEntropyLoss handles softmax internally)
    # It automatically ignores `ignore_index` (tgt_pad_idx)
    loss = loss_fn(logits_flat, target_out_flat)

    # Calculate accuracy metrics
    metrics = {'loss': loss.item()} # Use .item() to get scalar and detach
    with torch.no_grad(): # No need for gradients during accuracy calculation
        for k in [1, 5, 10]:
            # Pass logits and targets directly, ignore padding index
            acc_k = topk_accuracy(logits_flat, target_out_flat, k, ignore_index=tgt_pad_idx)
            metrics[f'top-{k}'] = acc_k.item()

    return metrics


def eval_model(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, config: dict) -> dict:
    """Evaluate the model on the given dataloader."""
    device = config['device']
    tgt_pad_idx = config['tgt_pad_idx']
    model_type = config.get('model_type', 'encoder_decoder') # Get model type from config
    logs = defaultdict(list)

    model.to(device)
    model.eval() # Set model to evaluation mode

    with torch.no_grad(): # Disable gradient calculations
        for source, target in tqdm(dataloader, desc="Evaluating"):
            # Calculate metrics for the batch
            metrics = loss_batch(model, source, target, loss_fn, tgt_pad_idx, device, model_type)
            for name, value in metrics.items():
                # Value is already a float scalar from loss_batch
                logs[name].append(value)

    # Average metrics over all batches
    avg_logs = {name: np.mean(values) for name, values in logs.items()}
    return avg_logs


def train_model(model: nn.Module, config: dict):
    """Train the model."""
    # Extract config parameters
    train_loader = config['train_loader']
    val_loader = config['val_loader']
    optimizer = config['optimizer']
    loss_fn = config['loss']
    epochs = config['epochs']
    device = config['device']
    clip = config.get('clip', None) # Gradient clipping value (optional)
    log_every = config['log_every']
    tgt_pad_idx = config['tgt_pad_idx']
    src_vocab = config['src_vocab']
    tgt_vocab = config['tgt_vocab']
    src_tokenizer = config['src_tokenizer']
    max_seq_len = config['max_sequence_length']
    model_type = config.get('model_type', 'encoder_decoder') # Get model type

    # WandB logging setup (assuming wandb.init is called outside)
    columns = ['epoch', 'train_source', 'train_target', 'train_predicted', 'train_likelihood',
               'val_source', 'val_target', 'val_predicted', 'val_likelihood']
    log_table = wandb.Table(columns=columns) if wandb.run else None

    print(f"Starting training for {epochs} epochs on {device}...")

    for e in range(epochs):
        model.train() # Set model to training mode
        epoch_logs = defaultdict(list)
        batch_iterator = tqdm(train_loader, desc=f"Epoch {e+1}/{epochs} Training")

        for batch_id, (source, target) in enumerate(batch_iterator):
            optimizer.zero_grad()

            # Calculate loss and other metrics for the batch
            metrics = loss_batch(model, source, target, loss_fn, tgt_pad_idx, device, model_type)
            loss = metrics['loss'] # Retrieve the loss tensor (before .item()) for backprop

            # Need to recalculate loss for backpropagation if loss_batch returns .item()
            # Re-run forward pass to get loss tensor:
            source, target = source.to(device), target.to(device)
            target_in = target[:, :-1]
            target_out = target[:, 1:]
            if model_type == 'encoder_decoder':
                 logits = model(source, target_in)
            else: # decoder_only
                 logits = model(target_in) # Adjust if needed
            logits_flat = logits.reshape(-1, logits.shape[-1])
            target_out_flat = target_out.reshape(-1)
            loss_tensor = loss_fn(logits_flat, target_out_flat) # Get the tensor loss

            # Backward pass and optimization step
            loss_tensor.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # Store metrics (use values returned by loss_batch which are already scalars)
            for name, value in metrics.items():
                epoch_logs[name].append(value)

            # Update progress bar description
            if batch_id % log_every == 0 or batch_id == len(train_loader) - 1:
                 avg_batch_loss = np.mean(epoch_logs['loss'][-log_every:])
                 batch_iterator.set_postfix(loss=f"{avg_batch_loss:.4f}")

            # Optional WandB logging per batch (can be verbose)
            # if batch_id % log_every == 0:
            #     wandb.log({f'Batch Train - {m}': v for m, v in metrics.items()})

        # --- End of Epoch ---
        # Calculate average training metrics for the epoch
        avg_train_logs = {name: np.mean(values) for name, values in epoch_logs.items()}
        print_logs('Train', avg_train_logs, e, epochs)
        wandb_train_logs = {f'Train - {m}': v for m, v in avg_train_logs.items()}

        # Evaluate on validation set
        val_logs = eval_model(model, val_loader, loss_fn, config)
        print_logs('Validation', val_logs, e, epochs)
        wandb_val_logs = {f'Validation - {m}': v for m, v in val_logs.items()}

        # Log combined metrics to WandB
        if wandb.run:
            wandb.log({**wandb_train_logs, **wandb_val_logs}, step=e+1) # Log per epoch

            # Log example predictions to WandB Table
            try:
                # Get random examples
                train_idx = torch.randint(len(train_loader.dataset), (1,)).item()
                val_idx = torch.randint(len(val_loader.dataset), (1,)).item()
                train_en_str, train_fr_str = train_loader.dataset.dataset[train_idx]
                val_en_str, val_fr_str = val_loader.dataset.dataset[val_idx]

                # Generate predictions using beam search
                train_preds = beam_search(
                    model, train_en_str, src_vocab, tgt_vocab, src_tokenizer,
                    'cpu', beam_width=3, max_target_sentences=1, max_sentence_length=max_seq_len, model_type=model_type
                )
                val_preds = beam_search(
                    model, val_en_str, src_vocab, tgt_vocab, src_tokenizer,
                    'cpu', beam_width=3, max_target_sentences=1, max_sentence_length=max_seq_len, model_type=model_type
                )

                train_pred_str, train_prob = train_preds[0] if train_preds else ("N/A", 0.0)
                val_pred_str, val_prob = val_preds[0] if val_preds else ("N/A", 0.0)

                log_table.add_data(
                    e + 1,
                    train_en_str, train_fr_str, train_pred_str, train_prob,
                    val_en_str, val_fr_str, val_pred_str, val_prob,
                )
            except Exception as ex:
                 print(f"Warning: Could not generate beam search predictions for logging. Error: {ex}")


    # Log the final table at the end of training
    if wandb.run and log_table is not None:
        wandb.log({'Model Predictions': log_table})
    print("Training finished.")
