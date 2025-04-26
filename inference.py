import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import Vocab
from itertools import takewhile
import einops # Make sure einops is imported

# --- Utility Functions ---
def beautify(sentence: str) -> str:
    """Removes useless spaces around punctuation."""
    punc = {'.', ',', ';', ':', '!', '?'}
    links = {'-', "'"} # Apostrophe included

    # Remove space before punctuation
    for p in punc:
        sentence = sentence.replace(f' {p}', p)

    # Remove space around linking characters (like apostrophe in l'arbre)
    for l in links:
        sentence = sentence.replace(f'{l} ', l)
        sentence = sentence.replace(f' {l}', l)

    # Handle specific French contractions (optional, but improves readability)
    sentence = sentence.replace(" n ' t", "n't") # Example for English, adapt for French
    sentence = sentence.replace(" j ' ai", "j'ai")
    sentence = sentence.replace(" l ' ", "l'")
    # Add more French specific cases if needed

    return sentence.strip() # Remove leading/trailing spaces


def indices_terminated(target: torch.LongTensor, eos_token: int) -> tuple:
    """Split indices between terminated (contains EOS) and non-terminated sentences."""
    terminated_mask = torch.any(target == eos_token, dim=1)
    terminated_indices = torch.where(terminated_mask)[0]
    non_terminated_indices = torch.where(~terminated_mask)[0]
    return terminated_indices, non_terminated_indices


def append_beams(target: torch.LongTensor, beams: torch.LongTensor) -> torch.LongTensor:
    """Add beam tokens to current sentences, duplicating sentences."""
    batch_size, n_beams = beams.shape
    n_tokens = target.shape[1]

    # Repeat each sentence in the target tensor 'n_beams' times
    # target shape: [batch_size, n_tokens] -> [batch_size * n_beams, n_tokens]
    target_repeated = torch.repeat_interleave(target, repeats=n_beams, dim=0)

    # Reshape beams to be appended
    # beams shape: [batch_size, n_beams] -> [batch_size * n_beams, 1]
    beams_flat = beams.reshape(-1, 1)

    # Concatenate the repeated targets with the flattened beams
    # Result shape: [batch_size * n_beams, n_tokens + 1]
    target = torch.cat((target_repeated, beams_flat), dim=1)
    return target

# --- Beam Search Implementation ---

def beam_search(
        model: nn.Module,
        source_sentence: str, # Changed name for clarity
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        src_tokenizer,
        device: str,
        beam_width: int,
        max_target_sentences: int, # Renamed for clarity
        max_sentence_length: int,
        model_type: str = 'encoder_decoder' # Add model type flag
    ) -> list:
    """Performs beam search to generate translations.

    Args:
        model: The translation model (either TranslationTransformer or DecoderOnlyTranslationTransformer).
        source_sentence: The sentence string to translate.
        src_vocab: Source vocabulary.
        tgt_vocab: Target vocabulary.
        src_tokenizer: Source tokenizer function.
        device: Device for inference ('cuda' or 'cpu').
        beam_width: Number of top-k tokens to keep at each step.
        max_target_sentences: Maximum number of candidate sentences to maintain.
        max_sentence_length: Maximum token length for generated sentences.
        model_type: 'encoder_decoder' or 'decoder_only'.

    Returns:
        List of (sentence, probability) tuples, sorted by probability (descending).
    """
    model.eval() # Set model to evaluation mode
    model.to(device)

    BOS_IDX = tgt_vocab['<bos>']
    EOS_IDX = tgt_vocab['<eos>']
    PAD_IDX = tgt_vocab['<pad>'] # Needed for decoder-only potentially

    # 1. Prepare source input (common for both model types)
    src_tokens = [src_vocab['<bos>']] + src_vocab(src_tokenizer(source_sentence)) + [src_vocab['<eos>']]
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device) # [1, src_len]

    # 2. Initialize target sequences and probabilities
    # Start with only the BOS token
    live_tgt_tokens = torch.LongTensor([[BOS_IDX]]).to(device) # [1, 1]
    live_tgt_probs = torch.FloatTensor([1.0]).to(device) # [1] - log prob starts at 0, prob at 1

    # Store completed sequences and their probabilities
    completed_tgt_tokens = []
    completed_tgt_probs = []

    with torch.no_grad():
        # 3. Main beam search loop
        for _ in range(max_sentence_length):
            if live_tgt_tokens.numel() == 0: # Stop if no live sequences left
                 break

            current_batch_size = live_tgt_tokens.shape[0]

            # Prepare model inputs based on type
            if model_type == 'encoder_decoder':
                # Repeat source for each live target sequence
                src_input = src_tensor.repeat(current_batch_size, 1) # [current_bs, src_len]
                # Target input is the current live sequences
                tgt_input = live_tgt_tokens # [current_bs, current_tgt_len]
                # Get model predictions (logits)
                logits = model(src_input, tgt_input) # Output: [current_bs, current_tgt_len, vocab_size]

            elif model_type == 'decoder_only':
                 # For decoder-only, the input sequence contains potentially both source and target parts
                 # Here, we assume the model expects the target sequence directly for generation
                 # (or a concatenation if trained that way - adjust if needed)
                 # We use live_tgt_tokens as the input sequence
                 logits = model(live_tgt_tokens) # Output: [current_bs, current_tgt_len, vocab_size]

            else:
                raise ValueError("Invalid model_type specified. Use 'encoder_decoder' or 'decoder_only'.")


            # Get probabilities for the *next* token only
            next_token_logits = logits[:, -1, :] # [current_bs, vocab_size]
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1) # Use log probs for stability

            # Calculate cumulative log probabilities for top-k next tokens
            # Add current sequence log probs (broadcast) to next token log probs
            # live_tgt_probs needs to be log probs now
            if _ == 0: # First step, convert initial prob 1 to log prob 0
                 live_tgt_log_probs = torch.log(live_tgt_probs)
            # Reshape probs: [current_bs, 1] + [current_bs, vocab_size] -> [current_bs, vocab_size]
            cumulative_log_probs = live_tgt_log_probs.unsqueeze(1) + next_token_log_probs

            # Find top `beam_width` candidates across all current sequences and possible next tokens
            # Flatten scores and indices: [current_bs * vocab_size]
            # We want the top `beam_width` overall highest probability sequences
            # If current_batch_size * beam_width > max_target_sentences, we might prune earlier,
            # but standard beam search selects top k *from each* live sequence first.
            # Let's keep beam_width from each. Total candidates = current_batch_size * beam_width
            top_k_log_probs, top_k_indices = torch.topk(
                cumulative_log_probs,
                k=beam_width,
                dim=1 # Get top k for each sequence in the batch
            ) # top_k_log_probs/indices: [current_bs, beam_width]

            # Expand beams: Create new candidate sequences and their probabilities
            # Indices need mapping back: top_k_indices are vocab indices
            # Original sequence index = floor(candidate_index / vocab_size) - not needed with dim=1 topk
            # Next token index = top_k_indices % vocab_size - directly given by top_k_indices

            new_live_tgt_tokens_list = []
            new_live_tgt_log_probs_list = []
            current_live_indices = torch.arange(current_batch_size, device=device) # [0, 1, ..., current_bs-1]

            for i in range(beam_width):
                next_tokens = top_k_indices[:, i] # [current_bs]
                next_log_probs = top_k_log_probs[:, i] # [current_bs]

                # Append next token to corresponding live sequences
                # live_tgt_tokens [current_bs, current_len] -> need to select based on original index
                # next_tokens [current_bs]
                current_seqs = live_tgt_tokens # No selection needed yet
                appended_seqs = torch.cat([current_seqs, next_tokens.unsqueeze(1)], dim=1) # [current_bs, current_len+1]

                new_live_tgt_tokens_list.append(appended_seqs)
                new_live_tgt_log_probs_list.append(next_log_probs)

            # Combine candidates from all beams
            live_tgt_tokens = torch.cat(new_live_tgt_tokens_list, dim=0) # [current_bs * beam_width, current_len+1]
            live_tgt_log_probs = torch.cat(new_live_tgt_log_probs_list, dim=0) # [current_bs * beam_width]

            # Prune: Keep only the top `max_target_sentences` overall
            if live_tgt_tokens.shape[0] > max_target_sentences:
                 top_overall_log_probs, top_overall_indices = torch.topk(
                     live_tgt_log_probs,
                     k=max_target_sentences
                 )
                 live_tgt_tokens = live_tgt_tokens[top_overall_indices]
                 live_tgt_log_probs = top_overall_log_probs # Already sorted

            # Separate completed sequences (ending in EOS)
            has_eos = (live_tgt_tokens == EOS_IDX).any(dim=1) # Check if EOS is present anywhere
            is_eos_last = (live_tgt_tokens[:, -1] == EOS_IDX) # Check specifically if the *last* token is EOS

            completed_mask = is_eos_last
            live_mask = ~completed_mask

            # Add completed sequences to the final list
            completed_tgt_tokens.append(live_tgt_tokens[completed_mask])
            completed_tgt_probs.append(torch.exp(live_tgt_log_probs[completed_mask])) # Convert back to probs

            # Keep only live sequences for the next iteration
            live_tgt_tokens = live_tgt_tokens[live_mask]
            live_tgt_log_probs = live_tgt_log_probs[live_mask]

            # Update current batch size for pruning logic if needed (or remove completed ones earlier)
            # Handled by filtering `live_tgt_tokens`

    # 4. After loop: Add any remaining live sequences to completed (they reached max length)
    if live_tgt_tokens.numel() > 0:
        completed_tgt_tokens.append(live_tgt_tokens)
        completed_tgt_probs.append(torch.exp(live_tgt_log_probs))

    # 5. Format results
    final_sentences = []
    final_probs = []
    if completed_tgt_tokens: # Check if list is not empty
        all_completed_tokens = torch.cat(completed_tgt_tokens, dim=0)
        all_completed_probs = torch.cat(completed_tgt_probs, dim=0)

        # Sort all completed sequences by probability
        sorted_probs, sorted_indices = torch.sort(all_completed_probs, descending=True)
        sorted_tokens = all_completed_tokens[sorted_indices]

        # Convert tokens to sentences
        for tokens, prob in zip(sorted_tokens, sorted_probs):
            token_list = tokens.tolist()
            # Remove BOS and truncate at EOS
            if BOS_IDX in token_list:
                start_idx = token_list.index(BOS_IDX) + 1
            else:
                start_idx = 0
            try:
                end_idx = token_list.index(EOS_IDX)
            except ValueError:
                end_idx = len(token_list) # Use full length if no EOS

            final_tokens = token_list[start_idx:end_idx]
            sentence = beautify(' '.join(tgt_vocab.lookup_tokens(final_tokens)))
            final_sentences.append(sentence)
            final_probs.append(prob.item()) # Store probability

    return list(zip(final_sentences, final_probs))
