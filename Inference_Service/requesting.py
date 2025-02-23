import requests

url = "http://192.168.0.103:8080/inference"
data = { "prompt": "Tell me a story ..." }
response = requests.post(url, json=data)

print("Response Status:", response.status_code)
print("Response JSON:", response.json())
