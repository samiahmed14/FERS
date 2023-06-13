import pandas as pd
import numpy as np
from model import create_convolutional_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def main():
    # Read data from csv file
    data = pd.read_csv("fer2013.csv")

    labels = data.iloc[:,[0]].values
    pixels = data['pixels']

    # Facial Expressions
    Expressions = {
        0:"Angry",
        1:"Disgust",
        2:"Fear",
        3:"Happy",
        4:"Sad",
        5:"Surprise",
        6:"Neutral"
    }

    labels = to_categorical(labels,len(Expressions))

    #converting pixels to Gray Scale images of 48X48 
    images = np.array([np.fromstring(pixel, dtype=int, sep=" ")for pixel in pixels])
    images = images/255.0
    images = images.reshape(images.shape[0],48,48,1).astype('float32')

    classes = 7
    model = create_convolutional_model(classes)
    print(model.summary())

    # Splitting data into training and test data
    train_images,test_images,train_labels,test_labels = train_test_split(images,labels,test_size=0.2,random_state=0)

    # Train the CNN
    model.fit(train_images,train_labels,batch_size=105,epochs=30,verbose=2)

if __name__ == "__main__":
    main()