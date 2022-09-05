# import the opencv library
import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.model.load_mode1('keras_moedel.h5')  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
    img = cv2.resize*(frame,(224,224))
    test = np.array(img,dtype=np.float32)
    test = np.expand_dims(test,axis=0)
    normalized = test/255
    normalized = test/255
    prediction=model.predict(normalized)
    print(prediction)
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()