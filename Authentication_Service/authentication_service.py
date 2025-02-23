from flask import Flask, request, jsonify
import os

app = Flask(__name__)
USER_FILE = "users.txt"

# Ensure user file exists
if not os.path.exists(USER_FILE):
    open(USER_FILE, "w").close()

def load_users():
    users = {}
    with open(USER_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                username, password = line.split(":", 1)
                users[username] = password
    return users

def save_user(username, password):
    with open(USER_FILE, "a") as f:
        f.write(f"{username}:{password}\n")

@app.route('/')
def root():
    return '''<h1>Hello from Home authentication</h1>'''

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({"message": "Username and password cannot be empty!"}), 400

    users = load_users()

    if username in users:
        return jsonify({"message": "Username already exists!"}), 409

    save_user(username, password)
    return jsonify({"message": "Signup successful!"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    users = load_users()

    if username in users and users[username] == password:
        return jsonify({"message": "Login successful!", "user": username})
    else:
        return jsonify({"message": "Invalid username or password!"}), 401

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)  # Runs on a different port
