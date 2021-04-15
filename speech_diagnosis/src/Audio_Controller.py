from Audio_Processing.feature_extraction import Feature_extraction
from Prediction.Predict import Predict

class Audio_Controller:

    def __init__(self, audio_file_location):

        self.audio_file_location = audio_file_location
        self.features_extraction = None
        