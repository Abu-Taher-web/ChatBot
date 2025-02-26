from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

app = Flask(__name__)

# Sample chatbot response logic
def chatbot_response(message):
    responses = {
        "hello": "Hi there! How can I help you?",
        "how are you": "I'm just a bot, but I'm doing fine!",
        "bye": "Goodbye! Have a great day!"
    }
    return responses.get(message.lower(), "I'm sorry, I don't understand.")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    bot_reply = chatbot_response(user_message)
    return jsonify({"response": bot_reply})


#Integrating Real AI
# Load GPT-2 model for text generation
# 

chatbot = pipeline("text-generation", model="gpt2")

def generate_gpt2_response(user_input):
    response = chatbot(user_input, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

@app.route('/gpt2', methods=['POST'])
def chat_gpt2():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"response": "Please enter a valid message."})

    bot_reply = generate_gpt2_response(user_message)
    return jsonify({"response": bot_reply})

# Load LLaMA model for text generation
#login("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")

# Load the model
'''model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

def generate_llama_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/llama', methods=['POST'])
def chat_llama():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"response": "Please enter a valid message."})

    bot_reply = generate_llama_response(user_message)
    return jsonify({"response": bot_reply})
'''


@app.route('/home')
def home():
    return '''<h1>Hello from inference service</h1>'''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
    #inference_service
    # python inference_service.py
    # cd Inference_Service
