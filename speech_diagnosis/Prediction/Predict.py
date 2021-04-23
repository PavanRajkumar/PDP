from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import json
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Predict:

    location = {
        "Scalar": "D:/Projects/Final-Year-Project/PDP/speech_diagnosis/ML_Models/Scalar/scalar.pkl",
        "NN": {"weights": "D:/Projects/Final-Year-Project/PDP/speech_diagnosis/ML_Models/NN/Speech_nn.h5", "architecture": "D:/Projects/Final-Year-Project/PDP/speech_diagnosis/ML_Models/NN/model_nn.json"},
        "KNN": {"model": "D:/Projects/Final-Year-Project/PDP/speech_diagnosis/ML_Models/KNN/KNN_model.pkl"},
        "RF": {"model": "D:/Projects/Final-Year-Project/PDP/speech_diagnosis/ML_Models/RF/RF_model.pkl"},
        "SVC": {"model": "D:/Projects/Final-Year-Project/PDP/speech_diagnosis/ML_Models/SVC/SVC_model.pkl"}
    }

    def __init__(self,model_name="NN"):

        self.model_name = model_name.upper()
        self.model = None
        self.scalar = None

        self.load_scalar()

        if self.model_name=="NN":
            self.model_load_success = self.load_nn_model(model_location=self.location[self.model_name])
        elif self.model_name.upper() in ["RF", "SVC", "KNN"]:
            self.model_load_success = self.load_other_models(self.location[self.model_name])


    def load_scalar(self):

        try:
            with open(self.location["Scalar"], "rb") as scalar_file:
                self.scalar = pickle.load(scalar_file)
        except Exception as e:
            print(e)
            return False
        return  True


    def load_other_models(self, model_location):
        try:
            with open(model_location["model"], "rb") as model_file:
                self.model = pickle.load(model_file)
        except Exception as e:
            print(e)
            return False
        return True


    def load_nn_model(self, model_location):

        try:
            print("LOAD_NN", model_location)
            with open(model_location["architecture"], "r") as json_file:
                loaded_model_json =  json_file.read()

            print("AFTER")

            self.model = model_from_json(loaded_model_json)
            self.model.load_weights(model_location["weights"])
            print("HELLO   ", self.model)
        except Exception as e:
            print(e)
            return False

        return True


    def clean_input(self, data):
        l=len(data.shape)

        if l==2:
            if not data.shape == (1,26):
                raise Exception("The dimensions of parameter do not match")
        elif l==1 :
            data = data.to_numpy().reshape(1, 26)
        else:
            raise Exception("The dimensions of parameter do not match")
        return data


    def scale_input(self, input):
        input = self.scalar.transform(input)
        return input


    def get_prediction(self, data):

        if not self.model_load_success:
            raise Exception("Model did not load successfully", self.model_load_success)

        data = np.array(data)

        try:
            input = self.clean_input(data)
            input= self.scale_input(input)
        except Exception as e:
            raise e

        prediction = self.model.predict(input).reshape(1,)[0]

        predictions_rounded = np.round_(prediction)

        diagnosis = (1==predictions_rounded)

        return {"Diagnosis": diagnosis, "probability": float("{0:.2f}".format(prediction*100))}


    def get_PR_curve(self):
        pass

    def get_ROC_curve(self):
        pass

    def get_lime_analysis(self):
        pass


if __name__ == "__main__":

    # predict = Predict("NN")
    data = [[1.488000e+00, 9.021300e-05, 9.000000e-01, 7.940000e-01,
        2.699000e+00, 8.334000e+00, 7.790000e-01, 4.517000e+00,
        4.609000e+00, 6.802000e+00, 1.355100e+01, 9.059050e-01,
        1.191160e-01, 1.113000e+01, 1.665330e+02, 1.647810e+02,
        1.042100e+01, 1.422290e+02, 1.875760e+02, 1.600000e+02,
        1.590000e+02, 6.064725e-03, 4.162760e-04, 0.000000e+00,
        0.000000e+00, 0.000000e+00]]

    # res = predict.get_prediction(data)

    predict_models = []
    res = []

    available_models = ["NN", "RF", "knn", "SVC"]
    for model in available_models:
        predict_models.append(Predict(model))

    for pred_mod in predict_models:
        res.append(pred_mod.get_prediction(data))

    for i in range(len(res)):
        print(res[i], available_models[i])
