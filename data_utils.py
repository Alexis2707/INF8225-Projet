import os
import zipfile
import requests
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator, Vocab
import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm # Added tqdm for potential use in download/processing

# Define special tokens globally or pass them as arguments
SPECIALS = ['<unk>', '<pad>', '<bos>', '<eos>']

# --- Data Download and Loading ---
def download_and_extract_data(url="http://www.manythings.org/anki/fra-eng.zip", target_dir="."):
    """Downloads and extracts the translation data."""
    zip_path = os.path.join(target_dir, "fra-eng.zip")
    data_path = os.path.join(target_dir, "fra.txt")

    if not os.path.exists(data_path):
        print(f"Downloading data from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an error for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(zip_path, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
             print("ERROR, something went wrong during download")

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("Extraction complete.")
        # os.remove(zip_path) # Optional: remove zip file after extraction
    else:
        print("Data file 'fra.txt' already exists.")

    df = pd.read_csv(data_path, sep='\\t', names=['english', 'french', 'attribution'])
    data_pairs = list(zip(df['english'], df['french']))
    return data_pairs

# --- Tokenizers ---
# Load spacy models (consider doing this once globally)
try:
    en_nlp = spacy.load('en_core_web_sm')
    fr_nlp = spacy.load('fr_core_news_sm')
except OSError:
    print("Downloading spaCy models...")
    os.system("python -m spacy download en_core_web_sm")
    os.system("python -m spacy download fr_core_news_sm")
    en_nlp = spacy.load('en_core_web_sm')
    fr_nlp = spacy.load('fr_core_news_sm')


def en_tokenizer(text):
    return [tok.text.lower() for tok in en_nlp.tokenizer(text)]

def fr_tokenizer(text):
    return [tok.text.lower() for tok in fr_nlp.tokenizer(text)]


# --- Vocabulary and Preprocessing ---

def yield_tokens(dataset, tokenizer, lang):
    """Tokenize the whole dataset and yield the tokens."""
    assert lang in ('en', 'fr')
    sentence_idx = 0 if lang == 'en' else 1

    for sentences in dataset:
        sentence = sentences[sentence_idx]
        tokens = tokenizer(sentence)
        yield tokens


def build_vocab(dataset: list, en_tokenizer, fr_tokenizer, min_freq: int):
    """Return two vocabularies, one for each language."""
    print("Building English vocabulary...")
    en_vocab = build_vocab_from_iterator(
        yield_tokens(dataset, en_tokenizer, 'en'),
        min_freq=min_freq,
        specials=SPECIALS,
    )
    en_vocab.set_default_index(en_vocab['<unk>'])  # Default token for unknown words
    print(f"English vocabulary size: {len(en_vocab):,}")

    print("Building French vocabulary...")
    fr_vocab = build_vocab_from_iterator(
        yield_tokens(dataset, fr_tokenizer, 'fr'),
        min_freq=min_freq,
        specials=SPECIALS,
    )
    fr_vocab.set_default_index(fr_vocab['<unk>'])
    print(f"French vocabulary size: {len(fr_vocab):,}")

    return en_vocab, fr_vocab


def preprocess(
        dataset: list,
        en_tokenizer,
        fr_tokenizer,
        max_words: int,
    ) -> list:
    """Preprocess the dataset."""
    filtered = []
    print(f"Preprocessing dataset (filtering sequences longer than {max_words} tokens)...")
    for en_s, fr_s in tqdm(dataset):
        # Simple length check before tokenization for speed
        # This is an approximation, actual token count might differ slightly
        if len(en_s.split()) > max_words * 1.5 or len(fr_s.split()) > max_words * 1.5:
             continue

        en_toks = en_tokenizer(en_s)
        fr_toks = fr_tokenizer(fr_s)

        if len(en_toks) >= max_words or len(fr_toks) >= max_words:
            continue

        en_s = en_s.replace('\\n', '').strip()
        fr_s = fr_s.replace('\\n', '').strip()

        filtered.append((en_s, fr_s))
    print(f"Filtered dataset size: {len(filtered):,}")
    return filtered

# --- Dataset Class ---

class TranslationDataset(Dataset):
    def __init__(
            self,
            dataset: list,
            en_vocab: Vocab,
            fr_vocab: Vocab,
            en_tokenizer,
            fr_tokenizer,
        ):
        super().__init__()

        self.dataset = dataset
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.en_tokenizer = en_tokenizer
        self.fr_tokenizer = fr_tokenizer
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.bos_idx = self.fr_vocab[self.bos_token] # Assuming target is French
        self.eos_idx = self.fr_vocab[self.eos_token]

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        """Return a sample."""
        en_sentence, fr_sentence = self.dataset[index]

        en_tokens = [self.bos_token] + self.en_tokenizer(en_sentence) + [self.eos_token]
        fr_tokens = [self.bos_token] + self.fr_tokenizer(fr_sentence) + [self.eos_token]

        en_indices = self.en_vocab(en_tokens)
        fr_indices = self.fr_vocab(fr_tokens)

        return torch.LongTensor(en_indices), torch.LongTensor(fr_indices)


# --- DataLoader Functions ---

def generate_batch(data_batch: list, src_pad_idx: int, tgt_pad_idx: int) -> tuple:
    """Add padding to the given batch."""
    en_batch, fr_batch = [], []
    for en_tokens, fr_tokens in data_batch:
        en_batch.append(en_tokens)
        fr_batch.append(fr_tokens)

    en_batch = pad_sequence(en_batch, padding_value=src_pad_idx, batch_first=True)
    fr_batch = pad_sequence(fr_batch, padding_value=tgt_pad_idx, batch_first=True)
    return en_batch, fr_batch


def build_dataloaders(
        max_sequence_length: int,
        min_token_freq: int,
        batch_size: int,
        data_url="http://www.manythings.org/anki/fra-eng.zip",
        data_dir=".",
        test_size=0.1,
        random_state=0
    ) -> tuple:
    """Builds and returns train/validation dataloaders and vocabularies."""

    # Download and load data
    all_data = download_and_extract_data(data_url, data_dir)

    # Split data
    train_data, val_data = train_test_split(all_data, test_size=test_size, random_state=random_state)
    print(f"\nRaw data sizes - Train: {len(train_data):,}, Validation: {len(val_data):,}")

    # Preprocess (filter long sequences)
    train_data_proc = preprocess(train_data, en_tokenizer, fr_tokenizer, max_sequence_length)
    val_data_proc = preprocess(val_data, en_tokenizer, fr_tokenizer, max_sequence_length)

    # Build vocabularies based on training data only
    en_vocab, fr_vocab = build_vocab(train_data_proc, en_tokenizer, fr_tokenizer, min_token_freq)

    # Create Dataset objects
    train_dataset = TranslationDataset(train_data_proc, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer)
    val_dataset = TranslationDataset(val_data_proc, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer)

    print(f'Processed dataset sizes - Training: {len(train_dataset):,}, Validation: {len(val_dataset):,}')

    # Padding indices
    src_pad_idx = en_vocab['<pad>']
    tgt_pad_idx = fr_vocab['<pad>']

    # Create DataLoader objects
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: generate_batch(batch, src_pad_idx, tgt_pad_idx)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        collate_fn=lambda batch: generate_batch(batch, src_pad_idx, tgt_pad_idx)
    )

    return train_loader, val_loader, en_vocab, fr_vocab, src_pad_idx, tgt_pad_idx
