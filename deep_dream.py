# -*- coding: utf-8 -*-

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
from argparse import ArgumentParser
import imutils
import tensorflow as tf
import numpy as np
import cv2


def loadImage(imagePath, width = 350):
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=width)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    return image

def deProcess(image):
    # to "undo" the processing done for Inception and cast the pixel 
    # values to integers
    
    image = 255 * (image + 1.0)
    image /= 2.0
    image = tf.cast(image, tf.uint8)
    
    return image

def calculteLoss(image, model):
    # add a batch dimension to the image and grab the activation from
    # specified layers of the Inception network after performing
    # a forward pass
    image = tf.expand_dims(image, axis=0)
    layerActivations = model(image)
    
    # initialize a list to store intermediate losses
    losses = []
    
    # iterate of the layer activation
    for act in layerActivations:
        # compute the mean of each activation and append it to the
		# losses list list
        loss = tf.reduce_mean(act)
        losses.append(loss)
    
    # return the sum of losses
    return tf.reduce_sum(losses)

@tf.function
def DeepDream(model, image, stepSize, eps=1e-8):
    # instruct tensorflow to record the gradients
    with tf.GradientTape() as tape:
        # keep track of the image to calculate gradient and calculate
        # the loss yeilded by the model
        tape.watch(image)
        loss = calculteLoss(image, model)
    
    # calculate the gradients of loss with respect to the image
    # and normalize the data
    gradients = tape.gradient(loss, image)
    gradients /= tf.math.reduce_std(gradients) + eps
    
    # adjust the image with the normalised gradients and clips its
    # pixel value to the range [0,1]
    
    image = image + (gradients * stepSize)
    image = tf.clip_by_value(image, -1, 1)
    
    return (loss, image)

def runDeepDreamModel(model, image, iterations=100, stepSize=0.01):
    # preprocess the image for input to inception model
    image = preprocess_input(image)
    
    # loop for the given number of iterations
    for iteration in range(iterations):
        # employ the DeepDream model to retreive the loss along with
        # the updated image
        (loss, image) = DeepDream(model=model, image=image, stepSize=stepSize)
        
        # log the losses after fixed interval
        if iteration % 25 == 0:
            print("[INFO] iteration {}, loss {}".format(iteration, loss))
        
    return deProcess(image)

# Construct argument parser
ap = ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image...")
ap.add_argument("-o", "--output", required=True, help="Path to output dreamed image...")
args = vars(ap.parse_args())

# define the layers we are going to use for the DeepDream
names = ["mixed3", "mixed5"]

# Define the octave scale and number of octaves
# NOTE : Tweeking this value will create different output dreams
OCTAVE_SCALE = 1.3
NUM_OCTAVE = 3

# Load the input image
print("[INFO] Loading input image...")
origImage = loadImage(args["image"])

# Load the pretrained inception model from the disk
print("[INFO] Loading the inception model...")
baseModel = InceptionV3(include_top=False, weights="imagenet")

# Construct the dreaming model
layers = [baseModel.get_layer(name).output for name in names]
dreamModel = tf.keras.Model(inputs = baseModel.input, outputs=layers)

# Convert the image to a tensorflow constant for better performance,
# Grab the first two dimensions of the image and cast them to float
image = tf.constant(origImage)
baseShape = tf.cast(tf.shape(image)[:-1], tf.float32)

# loop over the number of octave resolution that we are going to generate
for n in range(NUM_OCTAVE):
    # Compute the spatial dimentions( i.e width and height)
    # for the curent octave and cast them to integers
    print("[INFO] Starting octave {}".format(n))
    newShape = tf.cast(baseShape * (OCTAVE_SCALE ** n), tf.int32)
    
    # resize the image with newly computed shape, convert it to its
    # numpy variant, and run it through DeepDream Model
    image = tf.image.resize(image, newShape).numpy()
    image = runDeepDreamModel(model= dreamModel, image=image,iterations=200, stepSize=0.001)

# Convert the final image to a  numpy array and save it to disk
finalImage = np.array(image)
Image.fromarray(finalImage).save(args["output"])
    
    
    

    
    