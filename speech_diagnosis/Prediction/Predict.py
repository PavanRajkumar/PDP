from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import json
import numpy as np
import pickle
import src.Message as Message


class Predict:
    location = {
        "NN": {"weights": "..//ML_Models//NN//Speech_nn.h5", "architecture": "..//ML_Models//NN//model_nn.json",
               "Scalar": "..//ML_Models//NN//scalar.pkl"},
        "KNN": "",
        "RF": "",
        "SVC": ""
    }
    def __int__(self, model_name="NN"):

        self.model_name = model_name

        self.model = None
        self.scalar = None
        self.model_load_success = self.initialise_model(self.location[self.model_name])

        self.message = Message()


    def initialise_model(self, model_location):

        try:

            with open(model_location["architecture"]) as json_file:
                loaded_model_json =  json.load(json_file)

            self.model = model_from_json(loaded_model_json)
            self.model.load_weights(model_location["weights"])

            with open(model_location["Scalar"]) as Scalar_file:
                self.scalar = pickle.load(Scalar_file)


        except Exception as e:
            print(e)
            return False

        finally:
            json_file.close()

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


    def get_prediction(self, data):

        data = np.array(data)
        try:
            input = self.clean_input(data)
        except Exception as e:
            raise e

        # if not self.model_load_success:
        #     raise Exception("Model did not load Successfully")

        prediction = self.model.predict(input)

        predictions_rounded = np.round_(prediction)
        diagnosis = 1==predictions_rounded

        return {"Diagnosis": diagnosis, "probability": prediction}


    def get_PR_curve(self):
        pass

    def get_ROC_curve(self):
        pass

    def get_lime_analysis(self):
        pass




if __name__ == "__main__":

    predict = Predict()
    data = [[0.09149866, 0.108709  , 0.1053558 , 0.05296783, 0.10535138,
        0.17893973, 0.25821238, 0.15878218, 0.05406642, 0.14204353,
        0.15878427, 0.79927695, 0.13493302, 0.37640227, 0.21973716,
        0.21236662, 0.03370787, 0.19335322, 0.19911871, 0.10738255,
        0.10678308, 0.40134728, 0.05714651, 0.        , 0.        ,
        0.        ]]
    res = predict.get_prediction(data)

    print(res)