#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import os

MODEL_NAME = os.getenv('MODEL_NAME', 'dino_dragon_10_0.899.tflite')


# loading model
interpreter = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite') # model_path to model already contained in the second pulled docker image
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# getting the test data
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

# preprocessing test data
def preprocess(img):
    prep_img = prepare_image(img, target_size=(150, 150))
    scaled_img = scale_image(prep_img)
    return scaled_img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.Resampling.NEAREST)
    return img

def scale_image(img):
    img = np.array(img, dtype='float32')/255
    img = np.resize(img, (1, 150, 150, 3))
    return img

# predict
def predict(url):
    img = download_image(url)
    img = preprocess(img)
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    preds = dict({'dragon': preds[0][0].tolist(), 'dino': 1-(preds[0][0].tolist())})
    return preds

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

#url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg'
