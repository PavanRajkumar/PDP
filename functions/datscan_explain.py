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


def datscan_explain(datscan_sample):
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

    #Import LIME
    import lime
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    import matplotlib.pyplot as plt

    # create lime ImageExplainer
    explainer = lime_image.LimeImageExplainer()

    # Choosing the image from X_test_prep set
    image = scan_sample[0].astype(np.uint8)

    # Apply LIME to the image
    print("APPLYING LIME, THIS IS A CPU INTENSIVE PROCESS AND WILL TAKE TIME")
    explanation = explainer.explain_instance(image, vgg16.predict, top_labels=2, hide_color=0, num_samples=1000)

    #Arguments for get_image_and_mask() method defined below

    temp, mask = explanation.get_image_and_mask(0, positive_only=True, num_features=5, hide_rest=False)

    # Alert! You need to delete the explained.jpg file everytime you want to run an aexplanation on a new image.
    # Or else you will get a file overwrite error. Need to think of a fix for this.
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask).astype(np.uint8))
    plt.imsave("./static/images/Datscan/explained.jpg",mark_boundaries(temp / 2 + 0.5, mask).astype(np.uint8))

    return