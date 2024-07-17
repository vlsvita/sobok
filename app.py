from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# 모델과 토크나이저 불러오기
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 챗봇 함수 정의
def chatbot(question):
    inputs = tokenizer.encode(question + tokenizer.eos_token, return_tensors='pt')
    reply_ids = model.generate(inputs, max_length=100)
    reply = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return reply

@app.route("/")
def mainpage():
    return render_template('main.html')

@app.route("/guide")
def guidepage():
    return render_template('guide.html')

@app.route("/data")
def data():
    return render_template('data.html')

@app.route("/community")
def community():
    return render_template('community.html')

@app.route('/chatbot')
def chatbotpage():
    return render_template('chatbot.html')

@app.route("/data/Digital")
def digital():
    return render_template('data/Digital.html')

@app.route("/data/Coway")
def koway():
    return render_template('data/Coway.html')

@app.route("/data/Policy")
def jeongchek():
    return render_template('data/Policy.html')

@app.route("/data/LowIncome")
def jeosodeuk():
    return render_template('data/LowIncome.html')

@app.route("/data/Energy")
def energy():
    return render_template('data/Energy.html')

@app.route("/data/Teenager")
def cheongsonyeon():
    return render_template('data/Teenager.html')

@app.route("/data/NorthKoreanDefector")
def talbookin():
    return render_template('data/NorthKoreanDefector.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.json['message']
    reply = chatbot(message)
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True, port=4000)

