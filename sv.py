from flask import Flask, render_template, request, session,redirect, url_for
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session encryption

@app.route('/')
def index():
    return render_template('test.html')
@app.route('/add_value', methods=['POST'])
def add_value():
    value = request.form['value']  # Get the value from the submitted form
    if 'data_list' not in session:
        session['data_list'] = []  # Create an empty list in the session if it doesn't exist
    session['data_list'].append(value)  # Append the new value to the list
    print(session['data_list'])
    return redirect(url_for('index'))
if __name__ == '__main__':
    app.run()
