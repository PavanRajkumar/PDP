def preprocess_imgs(set_name, img_size):
    # Some imports
    import keras
    from keras.models import load_model 
    from keras.applications.vgg16 import VGG16, preprocess_input
    import tensorflow as tf
    import numpy as np
    from numpy import asarray
    import cv2

    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def datscan_predict(datscan_sample):  
    # Some imports
    import keras
    from keras.models import load_model 
    from keras.applications.vgg16 import VGG16, preprocess_input
    import tensorflow as tf
    import numpy as np
    from numpy import asarray
    import cv2
    import math
    IMG_SIZE = (224,224)

    vgg16 = load_model('./notebooks/spect_trained_final_1.h5')

    X = []
    img = cv2.imread(datscan_sample)
    #img = cv2.imread('60.jpg')
    X.append(img)
    X = np.array(X)

    scan_sample = preprocess_imgs(set_name=X, img_size=IMG_SIZE)

    prediction = vgg16.predict(scan_sample)
    print(float(prediction))

    if(float(prediction) < 0.5):
        return False
    else:
        return True

