from flask import Flask
from model import predict_result
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Hello, WideBot!</h1>"


@app.route('/predict', methods=['POST'])
def predict():
    """"
    send the data in json format
    Return Json object that contains 
    'label': Label of the data
    """
    receiver = request.get_json()
    print(receiver)
    labels = predict_result(receiver['data'])
    return labels


if __name__ == '__main__':
    app.run(debug=True)
