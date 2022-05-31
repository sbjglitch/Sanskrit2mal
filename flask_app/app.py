from flask import Flask, render_template, redirect, url_for, request
from torch_utils import translator

app = Flask(__name__)

@app.route('/')
def hello(methods=['GET', 'POST']):
    error=None

    if request.method == "POST":
        customer_input = request.form['name']
        return redirect(url_for("predictios"), name=customer_input)
    else:
        return render_template('basic.html', error=error)

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():

    name = request.form.get('name')
    mesg = f'Translated Slogan {translator(name)}'
    return mesg



if __name__ ==  "__main__":
    app.run(debug=True)