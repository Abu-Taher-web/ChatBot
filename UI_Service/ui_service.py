from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import requests  # To call the authentication microservice
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

app = Flask(__name__)
app.secret_key = "supersecretkey"
AUTH_SERVICE_URL = "http://localhost:5001"  # Address of authentication microservice

# Sample chatbot response logic
def chatbot_response(message):
    responses = {
        "hello": "Hi there! How can I help you?",
        "how are you": "I'm just a bot, but I'm doing fine!",
        "bye": "Goodbye! Have a great day!"
    }
    return responses.get(message.lower(), "I'm sorry, I don't understand.")

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    response = requests.post(f"{AUTH_SERVICE_URL}/signup", json=data)
    return jsonify(response.json()), response.status_code

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    response = requests.post(f"{AUTH_SERVICE_URL}/login", json=data)
    
    if response.status_code == 200:
        session['user'] = data['username']
    
    return jsonify(response.json()), response.status_code

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('root'))

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    bot_reply = chatbot_response(user_message)
    return jsonify({"response": bot_reply})
'''
# Integrating AI Chatbot with GPT-2
chatbot_gpt2 = pipeline("text-generation", model="gpt2")

def generate_gpt2_response(user_input):
    response = chatbot_gpt2(user_input, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

@app.route('/gpt2', methods=['POST'])
def chat_gpt2():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"response": "Please enter a valid message."})
    bot_reply = generate_gpt2_response(user_message)
    return jsonify({"response": bot_reply})
'''
# Load LLaMA model for text generation
# login("your_huggingface_token_here")
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

# def generate_llama_response(user_input):
#     inputs = tokenizer(user_input, return_tensors="pt")
#     outputs = model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# @app.route('/llama', methods=['POST'])
# def chat_llama():
#     data = request.json
#     user_message = data.get('message', '')
#     if not user_message:
#         return jsonify({"response": "Please enter a valid message."})
#     bot_reply = generate_llama_response(user_message)
#     return jsonify({"response": bot_reply})
'''
@app.route('/chatbot')
def chatbot():
    if 'user' not in session:
        return redirect(url_for('root'))
    return render_template('chatbot.html')
'''
@app.route('/')
def root():
    return render_template('chatbot.html')

@app.route('/home')
def home():
    return '''<h1>Hello from Home 123</h1>'''

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

# Run the UI service using: python ui_service.py
#cd UI_Service