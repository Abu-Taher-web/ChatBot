import os
import sys  # <-- Add this import

from flask import Flask, request, jsonify, render_template, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "supersecretkey"
USER_FILE = "users.txt"

# Ensure user file exists
if not os.path.exists(USER_FILE):
    open(USER_FILE, "w").close()

# Function to read users from file safely
def load_users():
    users = {}
    with open(USER_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:  # Ensure proper format
                username, password = line.split(":", 1)  # Split safely
                users[username] = password
    return users

# Function to save new user
def save_user(username, password):
    with open(USER_FILE, "a") as f:
        f.write(f"{username}:{password}\n")

# Signup route
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({"message": "Username and password cannot be empty!"}), 400

    users = load_users()

    if username in users:
        return jsonify({"message": "Username already exists. Choose a different one!"}), 409

    save_user(username, password)
    return jsonify({"message": "Signup successful! You can now log in."})

# Login route
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    users = load_users()

    if username in users and users[username] == password:
        session['user'] = username
        return jsonify({"message": "Login successful!", "user": username})
    else:
        return jsonify({"message": "Invalid username or password!"}), 401

# Logout route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('root'))

# Route for chatbot UI
@app.route('/chatbot')
def chatbot():
    if 'user' not in session:
        return redirect(url_for('root'))
    return render_template('chatbot.html')

# Chatbot logic
def chatbot_response(message):
    responses = {
        "hello": "Hi there! How can I help you?",
        "how are you": "I'm just a bot, but I'm doing fine!",
        "bye": "Goodbye! Have a great day!"
    }
    return responses.get(message.lower(), "I'm sorry, I don't understand.")

@app.route('/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify({"response": "Unauthorized. Please log in."}), 401

    data = request.json
    user_message = data.get('message', '')
    bot_reply = chatbot_response(user_message)
    return jsonify({"response": bot_reply})

@app.route('/')
def root():
    return render_template('index.html')

# ✅ **Add Port Handling Here**
if __name__ == '__main__':
    # Default port is 5000, but allow custom ports
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number. Using default port 5000.")

    app.run(host="0.0.0.0", port=port, debug=True)  # Now supports dynamic port numbers
