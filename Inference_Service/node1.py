import grpc
import pickle
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import inference_pb2
import inference_pb2_grpc

# ----------------------
# First-Half Inference Function
# ----------------------

def first_half_model(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    outputs = model.transformer(input_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = outputs.past_key_values
    return input_ids, attention_mask, past_key_values

# ----------------------
# Load Model and Tokenizer Locally
# ----------------------

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.to("cpu")
model.eval()

# ----------------------
# Prepare the Input
# ----------------------

prompt = "Tell me a short story within 100 words. Use simple terms. Do not repeat same sentences."
input_ids, attention_mask, past_key_values = first_half_model(prompt, model, tokenizer)

# Serialize the intermediate state using pickle.
serialized_input_ids = pickle.dumps(input_ids)
serialized_attention_mask = pickle.dumps(attention_mask)
serialized_past_key_values = pickle.dumps(past_key_values)

# ----------------------
# Create gRPC Client and Make RPC Call
# ----------------------

# Set client options to support large messages (here, 50 MB).
client_options = [
    ('grpc.max_send_message_length', 50 * 1024 * 1024),
    ('grpc.max_receive_message_length', 50 * 1024 * 1024),
]
<<<<<<< HEAD
channel = grpc.insecure_channel('localhost:50051', options=client_options)
=======
channel = grpc.insecure_channel('192.168.0.101:50051', options=client_options)
>>>>>>> cd1e8278c68c0ac7583b157c5cbaa82a8a606841
stub = inference_pb2_grpc.InferenceServiceStub(channel)

# Build the request message.
request = inference_pb2.IntermediateData(
    input_ids=serialized_input_ids,
    attention_mask=serialized_attention_mask,
    past_key_values=serialized_past_key_values
)

# Send the request and get the response.
response = stub.ContinueGeneration(request)
print("Generated Continuation:", response.continuation)

# python node1.py