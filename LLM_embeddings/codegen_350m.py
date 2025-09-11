from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load CodeGen model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi", output_hidden_states=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_context_embedding(codebase: str, method: str = "avg", max_tokens: int = 2048) -> torch.Tensor:
    """
    Generate code embedding using CodeGen (decoder-only causal model).

    Args:
        codebase (str): Input source code
        method (str): One of ['first', 'avg', 'last']
        max_tokens (int): Maximum tokens per chunk (<=2048 for codegen-350M)

    Returns:
        torch.Tensor: Embedding vector (shape: [hidden_dim])
    """
    tokens = tokenizer.tokenize(codebase)
    chunk_size = max_tokens - 1  # CodeGen doesn't use special start token
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    embeddings = []

    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # CodeGen is decoder-only: use hidden_states[-1]
            last_hidden = outputs.hidden_states[-1]  # shape: [1, seq_len, hidden_dim]

        attention_mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids'])).to(device)
        if method == "first":
            emb = last_hidden[:, 0, :]  # First token
        elif method == "avg":
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            emb = summed / counts
        elif method == "last":
            valid_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(inputs['input_ids'].size(0)).to(device)
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


