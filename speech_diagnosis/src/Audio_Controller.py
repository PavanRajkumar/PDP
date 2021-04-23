from speech_diagnosis.Audio_Processing.feature_extraction import Feature_extraction
from speech_diagnosis.Prediction.Predict import Predict

class Audio_Controller:

    def __init__(self, audio_file_location):

        self.audio_file_location = audio_file_location
        self.features_extraction = None
        self.features = None
        self.predict_object = None
        self.prediction = None


    def process_audio(self):

        self.features_extraction = Feature_extraction(self.audio_file_location)
        self.features = self.features_extraction.get_features_as_numpy()


    def predict_PD_diagnosis(self, model_name="NN"):

        self.predict_object = Predict(model_name=model_name)
        self.prediction = self.predict_object.get_prediction(data=self.features)

        return self.prediction


if __name__ == "__main__":
    audio = Audio_Controller("..\..\files\speech\muhammadali_parkinsondisease.mp3")
    audio.process_audio()
    res = audio.predict_PD_diagnosis()

    print(res)
