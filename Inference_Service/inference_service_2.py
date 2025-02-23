from flask import Flask, request, jsonify
import grpc
import pickle
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import inference__pb2
import inference__pb2_grpc

app = Flask(__name__)

# ------------------------------
# Load GPT-2 Model and Tokenizer
# ------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.to("cpu")
model.eval()

# ------------------------------
# Define the First-Half Inference
# ------------------------------
def first_half_model(input_text, model, tokenizer):
    """
    Processes the prompt and returns the intermediate data (input_ids,
    attention_mask, and past_key_values) for further generation.
    """
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    outputs = model.transformer(input_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = outputs.past_key_values
    return input_ids, attention_mask, past_key_values

# ------------------------------
# Inference Endpoint
# ------------------------------
@app.route('/inference', methods=['POST'])
def inference():
    # Expect a JSON payload like: {"prompt": "Your text here."}
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Run first half of inference.
    input_ids, attention_mask, past_key_values = first_half_model(prompt, model, tokenizer)
    
    # Serialize the intermediate results using pickle.
    serialized_input_ids = pickle.dumps(input_ids)
    serialized_attention_mask = pickle.dumps(attention_mask)
    serialized_past_key_values = pickle.dumps(past_key_values)

    # Configure gRPC client options to support large messages (e.g., 50 MB).
    client_options = [
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
    ]
    # Replace 'NODE2_IP_ADDRESS' with the actual IP address of Node 2.
    channel = grpc.insecure_channel('NODE2_IP_ADDRESS:50051', options=client_options)
    stub = inference__pb2_grpc.InferenceServiceStub(channel)

    # Build the gRPC request.
    request_proto = inference__pb2.IntermediateData(
        input_ids=serialized_input_ids,
        attention_mask=serialized_attention_mask,
        past_key_values=serialized_past_key_values
    )

    try:
        # Call Node 2 to complete the generation.
        response_proto = stub.ContinueGeneration(request_proto)
        generated_text = response_proto.continuation
    except grpc.RpcError as e:
        return jsonify({"error": f"gRPC error: {e}"}), 500

    return jsonify({
        "prompt": prompt,
        "generated_text": generated_text
    })

# ------------------------------
# Start the Flask App
# ------------------------------
if __name__ == '__main__':
    # The Flask app listens on port 8080.
    app.run(host='0.0.0.0', port=8080)
