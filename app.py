from flask import Flask, render_template, jsonify, request
from train import DataInitialiser
from nn import NeuralNetwork
import numpy as np 
import random, threading

app = Flask(__name__)

TOTAL_ITERATIONS = 1000

data = DataInitialiser()
nn = NeuralNetwork(alpha=0.1)

# Global variable to track training progress
training_progress = {
    'iteration': 0,
    'accuracy': 0.0
}
# For testing accuracy
# for i in range(1):
#     nn.test_prediction(i, data.X_train, data.y_train, plot_it=False, print_out=True)
#     print("HHHHHH", data.X_train[:, i, None].shape)

def train_model():
    global training_progress
    for i in range(TOTAL_ITERATIONS):
        nn.train(data.X_train, data.y_train, 1, print_out=False)  
        training_progress['iteration'] = i + 1
        training_progress['accuracy'] = nn.accuracy

threading.Thread(target=train_model).start()

@app.route("/")
def index():
    return render_template('index.html', prediction_result="??", total_iterations=TOTAL_ITERATIONS)

# TODO: allow for mnist images to be uploaded to the canvas to check that they correspond to what would be expected
@app.route("/predict", methods=["POST"])
def predict():
    if nn is not None:
        # get the data passed in 
        data = request.get_json()
        pixels = data['pixels'] 
        # make the right shape
        drawn_image_pixels_matrix = (np.array(pixels) / 255.0).reshape((784, 1))
        prediction_result, likelihoods = nn.guess_my_digit(drawn_image_pixels_matrix)
        return jsonify(prediction=str(prediction_result[0]), likelihoods=str(likelihoods)) #TODO: configure likehoods as graph 
    else:
        return jsonify(prediction="Model not trained")

@app.route("/get-mnist", methods=["GET"])
def loadMNIST():
    rand_num = random.randint(0, data.m -1 )
    pixels = (data.X_train[:, rand_num, None] * 255).astype(int)
    pixels_list = pixels.flatten().tolist()
    return jsonify(pixels_list)

@app.route("/get-progress", methods=["GET"])
def get_progress():
    return jsonify(training_progress)
    

if __name__ == '__main__':
    app.run(debug=True)