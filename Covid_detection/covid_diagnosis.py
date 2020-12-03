
import test
import numpy as np
import argparse
import cv2

DIAGNOSIS_MESSAGES = [ "Pneumonia detected", "Covid19 detected", "Normal lungs detected" ]

model=test.model
#Function written by Jordan to do online inference i.e. Covid19 tests
def doOnlineInference_covid19Pneumonia (imagePath):
    test_data = []
    img = test.cv2.imread(imagePath,0) #Replace plt.imread, with  gray scale cv2.imread(path,0), so that ui's image load process doesn't throw a pyimage10 error
    img = test.cv2.resize(img, (224, 224))
    img = test.np.dstack([img, img, img])
    img = img.astype('float32') / 255
    test_data.append(img)
    prediction = model.predict(np.array(test_data))
    _prediction = round( prediction[0][0]*100, 3 )
    if ( _prediction > 50 ):
        _prediction = DIAGNOSIS_MESSAGES[1];
    elif ( _prediction < 50 ):
        _prediction = DIAGNOSIS_MESSAGES[2];
    outputContent = _prediction + "\n"
    outputContent += "Raw Neural Network Output : " + str(prediction[0][0]) + ". A value closer to 1 signifies illness, while a value closer to 0 signifies normalness.\n\n"
    recordInferenceEvent (imagePath, outputContent)
    return outputContent



#Record each inference in a text file
import datetime
def recordInferenceEvent ( imagePath, outputContent ):
    currentDate = datetime.datetime.now()
    with open("inference_record.txt", "a") as text_file:
        text_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        text_file.write("DATE/TIME : " + str(currentDate.month) + " " + str(currentDate.day) + ", " + str(currentDate.year) + "..." + str(currentDate.hour) + ":" + str(currentDate.minute) + ":" + str(currentDate.second) + "\n\n")
        text_file.write("IMAGE : " + imagePath + "\n\n")
        text_file.write("RESULT : \n" + outputContent + "\n\n\n\n")