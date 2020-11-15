import os, sys, json
from flask import Flask, request, jsonify, make_response
from waitress import serve
import lrcnreg.scripts.preprocessing as prep
import lrcnreg.scripts.loading_data as ld
from lrcnreg.scripts.model import CNN, LRCN
import torch
from PIL import Image

    
apiprefix = ""

#Specify where model is located
MODEL_PATH = './model_weights.pt'
UPLOAD_FOLDER = './'

ALLOWED_EXTENSIONS = ['mp4', 'mov', 'flv', 'webm']


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = LRCN()
model.load_state_dict(torch.load(MODEL_PATH))
model.to('cpu')
model.eval() #Switching model to evaluation mode

def allowed_format(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("%s/status" % apiprefix, methods=['GET'])
def check():
    return 'OK'

@app.route("%s/predict" % apiprefix, methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"status": "Error", "info": "Part 'video' is missing in request"}), 400
    video = request.files['video']
    if video.filename == '':
        return jsonify({"status": "Error", "info": "Empty filename"}), 400
    if video and allowed_format(video.filename):
        filename = 'trailer.mp4'
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        score = model.predict_single("trailer.mp4")
        reply = {'status': 'Success', 'movie_score': score}
        return jsonify(reply), 200
    return jsonify({"status": "Error", "info": "Invalid file extension, allowed %s" % ','.join(str(s) for s in ALLOWED_EXTENSIONS)}), 400
    

if __name__ == "__main__":
    serve(app, host=os.getenv('APP_ADDR', '0.0.0.0'), port=int(os.getenv('APP_PORT', 5000)))
