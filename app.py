from flask import Flask, render_template, request
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        print('get method')
        return render_template('index.jinja')
    elif request.method == 'POST':
        if bool(request.form):
            data = request.form['text-input']
        elif bool(request.files):
            data = request.files['file-input']
        print(request.form)
        print(request.files)
        return render_template('index.jinja')
