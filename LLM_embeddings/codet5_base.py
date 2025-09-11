from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load CodeT5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_context_embedding(codebase: str, method: str = "avg", max_tokens: int = 512) -> torch.Tensor:
    """
    Generate code embedding using CodeT5 (encoder-decoder).

    Args:
        codebase (str): Input source code
        method (str): One of ['first', 'avg', 'last']
        max_tokens (int): Maximum tokens per chunk (should not exceed 512 for CodeT5-small)

    Returns:
        torch.Tensor: Embedding vector (shape: [hidden_dim])
    """
    tokens = tokenizer.tokenize(codebase)
    chunk_size = max_tokens - 2  # reserve <pad> and </s> (CodeT5 uses pad/eos)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    embeddings = []

    for chunk in chunks:
        chunk_ids = tokenizer.convert_tokens_to_ids(chunk)
        input_ids = torch.tensor([[tokenizer.pad_token_id] + chunk_ids + [tokenizer.eos_token_id]]).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            encoder = model.get_encoder()
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]

        if method == "first":
            emb = last_hidden[:, 0, :]  # first token embedding (usually <pad>)
        elif method == "avg":
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            emb = summed / counts
        elif method == "last":
            valid_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(input_ids.size(0)).to(device)
            emb = last_hidden[batch_idx, valid_lengths, :]
        else:
            raise ValueError("method must be one of ['first', 'avg', 'last']")

        embeddings.append(emb)

    final_embedding = torch.stack(embeddings, dim=0).mean(dim=0).squeeze(0)  # shape: [hidden_dim]

    return final_embedding


def test():
    embedding = get_context_embedding("def add(a, b): return a + b", method="first")
    print("Embedding shape:", embedding.shape)


if __name__ == "__main__":
    test()


