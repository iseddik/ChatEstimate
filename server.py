from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import json
import random
import backend
app = Flask(__name__)
app.secret_key = 'BAD_SECRET_KEY'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/i2a')
def i2a():
    if session.get('client') is None:
        return redirect(url_for('index'))
    print(session)
    return render_template('index.html')


@app.route('/')
def index():
    if (session.get('client') is None) or (session.get('server') is None) :
        session['client'] = []
        session['server'] = []
    return redirect(url_for('i2a'))


@app.route('/process_form', methods=['POST'])
def process_form():
    msg = request.form.get('name')
    serv = response_(msg)
    session["client"].append(msg)
    session["server"].append(serv)
    result = {'client': session["client"], 'server': session["server"]}
    return jsonify({'result': result})

def response_(msg):
    with open('intents.json', 'r') as file:
        data = json.load(file)
    for item in data['intents']:
        if msg.lower() in [ele.lower() for ele in item["patterns"]]:
            return random.choices(item['responses'], k=1)[0]

    return backend.feedback(msg)

if __name__ == '__main__':
    app.run(debug=True)
