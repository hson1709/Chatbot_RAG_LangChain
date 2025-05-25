from flask import Flask, render_template, request

from Chatbot.chain import detect_data_type, create_prompt_template, reranking
from Chatbot.conversation import create_conversational_chain, create_response, memory
from VectorDB.load_db import load_db
from Model.model import llm

app = Flask(__name__)

db_thuoc = load_db("thuoc")
db_benh = load_db("benh")
answer_prompt, classification_prompt = create_prompt_template()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form.get("msg")
    if not user_input:
        return "Xin hãy nhập câu hỏi.", 400
    data_type = detect_data_type(user_input, classification_prompt, llm)
    retriever = reranking(db_thuoc, db_benh, data_type)
    chain = create_conversational_chain(llm, retriever, memory, answer_prompt)
    response = create_response(user_input, chain)
    return response

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=8080)