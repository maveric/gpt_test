import os
from typing import Dict
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, jsonify
from flask import Flask
from .chat.chat import ChatSession

load_dotenv()
app = Flask(__name__)
# flask requires a secret to use sessions. 
app.secret_key = os.getenv("CHAT_APP_SECRET_KEY")
app.secret_key = "thisisasupersecretkey"


chat_sessions: Dict[str, ChatSession] = {}

@app.route("/")
def index():
    chat_session = _get_user_session()
    return render_template("chat.html", conversation=chat_session.get_messages())

def _get_user_session() -> ChatSession:
    """
    If a ChatSession exists for the current user return it
    Otherwise create a new session, add it into the session.
    """

    chat_session_id = session.get("chat_session_id")
    if chat_session_id:
        chat_session = chat_sessions.get(chat_session_id)
        if not chat_session:
            chat_session = ChatSession()
            chat_sessions[chat_session.session_id] = chat_session
            session["chat_session_id"] = chat_session.session_id
    else:
        chat_session = ChatSession()
        chat_sessions[chat_session.session_id] = chat_session
        session["chat_session_id"] = chat_session.session_id
    return chat_session

@app.route('/chat', methods=['POST'])
def chat():
    message: str = request.json['message']
    chat_session = _get_user_session()
    chatgpt_message = chat_session.get_chatgpt_response(message)
    return jsonify({"message": chatgpt_message})