from flask import Flask, request, jsonify, session
import requests
import itertools

app = Flask(__name__)

# List of chatbot instances
chatbot_instances = [
    "http://127.0.0.1:5000",
    "http://127.0.0.1:5001",
    "http://127.0.0.1:5002"
]

# Round-Robin iterator
chatbot_cycle = itertools.cycle(chatbot_instances)

# Load Balancer Route - Forwards chat requests to chatbot instances
@app.route('/chat', methods=['POST'])
def load_balancer():
    chatbot_url = next(chatbot_cycle)
    headers = {"Content-Type": "application/json"}

    if request.cookies.get("session"):
        headers["Cookie"] = f"session={request.cookies.get('session')}"  # Forward session cookie

    try:
        response = requests.post(f"{chatbot_url}/chat", json=request.json, headers=headers)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException:
        return jsonify({"error": "Chatbot instance unavailable"}), 503

# 🔹 **Fix: Add Login Route to Load Balancer**
@app.route('/login', methods=['POST'])
def login():
    chatbot_url = next(chatbot_cycle)
    try:
        response = requests.post(f"{chatbot_url}/login", json=request.json)

        # Forward session cookie to client if login successful
        if response.status_code == 200:
            resp = jsonify(response.json())
            if "Set-Cookie" in response.headers:
                resp.headers["Set-Cookie"] = response.headers["Set-Cookie"]
            return resp
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException:
        return jsonify({"error": "Authentication service unavailable"}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)
