import grpc
from concurrent import futures
import time
import pickle
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import inference_pb2
import inference_pb2_grpc

# ----------------------
# Utility Functions
# ----------------------

def top_k_top_p_filtering(logits, top_k=50, top_p=0.95, filter_value=-float('Inf')):
    """
    Filter logits using top-k and nucleus (top-p) filtering.
    """
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        logits = torch.where(logits < min_values, torch.full_like(logits, filter_value), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits

def no_repeat_ngram_filtering(generated, next_token_logits, ngram_size=4):
    """
    Blocks candidate tokens that would create a repeated n-gram.
    """
    generated_tokens = generated[0].tolist()
    if len(generated_tokens) < ngram_size - 1:
        return next_token_logits

    ngram_dict = {}
    for i in range(len(generated_tokens) - ngram_size + 1):
        ngram = tuple(generated_tokens[i:i+ngram_size])
        prefix = ngram[:-1]
        ngram_dict.setdefault(prefix, []).append(ngram[-1])
        
    current_prefix = tuple(generated_tokens[-(ngram_size - 1):])
    if current_prefix in ngram_dict:
        banned_tokens = ngram_dict[current_prefix]
        for token in banned_tokens:
            next_token_logits[0, token] = -float('Inf')
    return next_token_logits

def second_half_model(input_ids, attention_mask, past_key_values, model, tokenizer,
                      max_new_tokens=100, refresh_interval=10,
                      top_k=50, top_p=0.95, temperature=0.8,
                      repetition_penalty=1.0, ngram_size=4):
    """
    Continues generation using the cached past_key_values from the first half.
    """
    generated = input_ids
    model.eval()
    device = next(model.parameters()).device
    generated = generated.to(device)
    attention_mask = attention_mask.to(device)

    for i in range(max_new_tokens):
        if i > 0 and i % refresh_interval == 0:
            with torch.no_grad():
                outputs = model.transformer(
                    generated,
                    attention_mask=torch.ones_like(generated),
                    use_cache=True
                )
            past_key_values = outputs.past_key_values

        with torch.no_grad():
            outputs = model.transformer(
                generated[:, -1:],
                use_cache=True,
                past_key_values=past_key_values
            )
        logits = model.lm_head(outputs.last_hidden_state[:, -1, :]) / temperature
        past_key_values = outputs.past_key_values

        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                logits[0, token_id] /= repetition_penalty

        logits = no_repeat_ngram_filtering(generated, logits, ngram_size=ngram_size)
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode only the generated continuation.
    continuation = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return continuation

# ----------------------
# gRPC Service Definition
# ----------------------

class InferenceServiceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        # Load the GPT-2 model and tokenizer.
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to("cpu")
        self.model.eval()

    def ContinueGeneration(self, request, context):
        # Deserialize the incoming data using pickle.
        input_ids = pickle.loads(request.input_ids)
        attention_mask = pickle.loads(request.attention_mask)
        past_key_values = pickle.loads(request.past_key_values)
        
        # Perform the second half of generation.
        continuation = second_half_model(
            input_ids, attention_mask, past_key_values,
            self.model, self.tokenizer,
            max_new_tokens=100, refresh_interval=10,
            top_k=50, top_p=0.95, temperature=0.8,
            repetition_penalty=1.0, ngram_size=4
        )
        return inference_pb2.GenerationResult(continuation=continuation)

def serve():
    # Increase max message sizes to allow larger payloads (here, 50 MB).
    server_options = [
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=server_options)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051.")
    try:
        while True:
            time.sleep(86400)  # Keep the server alive.
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
    # python node2.py
