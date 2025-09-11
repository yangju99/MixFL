from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pdb

# Load GraphCodeBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModelForMaskedLM.from_pretrained("microsoft/graphcodebert-base")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_context_embedding(codebase: str, method: str = "avg", max_tokens: int = 512) -> torch.Tensor:
    """
    Generate code embedding using GraphCodeBERT (encoder-only).

    Args:
        codebase (str): Input source code
        method (str): One of ['cls', 'avg', 'last']
        max_tokens (int): Maximum tokens per chunk

    Returns:
        torch.Tensor: Embedding vector (shape: [hidden_dim])
    """
    tokens = tokenizer.tokenize(codebase)
    chunk_size = max_tokens - 2  # reserve space for [CLS] and [SEP]
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

    embeddings = []

    for chunk in chunks:
        chunk_ids = tokenizer.convert_tokens_to_ids(chunk)
        input_ids = torch.tensor([[tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]]).to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]

        if method == "cls":
            if hasattr(tokenizer, "cls_token_id"):
                emb = last_hidden[:, 0, :]  # CLS token
            else:
                raise ValueError("Model/tokenizer does not support [CLS] token. Use 'avg' or 'last'.")
        elif method == "avg":
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            emb = summed / counts
        elif method == "last":
            valid_lengths = attention_mask.sum(dim=1) - 1  # position of last token
            batch_idx = torch.arange(input_ids.size(0)).to(device)
            emb = last_hidden[batch_idx, valid_lengths, :]
        else:
            raise ValueError("method must be one of ['cls', 'avg', 'last']")

        embeddings.append(emb)

    final_embedding = torch.stack(embeddings, dim=0).mean(dim=0).squeeze(0)  # shape: [hidden_dim]
    
    return final_embedding


def test():
    embedding = get_context_embedding("def add(a, b): return a + b", method="cls")
    print("Embedding shape:", embedding.shape)


if __name__ == "__main__":
    test()
    

