import parselmouth
from parselmouth.praat import call
import glob
import pandas as pd
import numpy as np

class Feature_extraction:

    def __init__(self):

        self.speech_parameters = {
            "Pitch": None,
            "Mean_Pitch": None,
            "Std_dev":None,
            "Harmonicity":None,
            "HTN":None,
            "NTH": None,
            "Jitter_Local":None,
            "Jitter_Absolute_Local": None,
            "Jitter_Rap": None,
            "Jitter_ppq5":None,
            "Jitter_ddp": None,
            "Shimmer_Local": None,
            "Shimmer_Local_db":None,
            "Shimmer_apq3" : None,
            "Shimmer_aqpq5" : None,
            "Shimmer_apq11": None,
            "Shimmer_dda": None,

        }

    def measurePitch(self, voiceID, f0min, f0max, unit):

        sound = parselmouth.Sound(voiceID) # read the sound

        #Pitch
        pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
        self.speech_parameters["Pitch"] = pitch

        self.speech_parameters["Mean_Pitch"] = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
        self.speech_parameters["Std_dev"] = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation

        #Harmonicity
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        self.speech_parameters["Harmonicity"] = harmonicity
        self.speech_parameters["HNR"] = call(harmonicity, "Get mean", 0, 0)

        #Jitter
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

        #Shimmer
        localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        return meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer

    file_list = []
    mean_F0_list = []
    sd_F0_list = []
    hnr_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    ddpJitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    ddaShimmer_list = []



    # Go through all the wave files in the folder and measure pitch
    for wave_file in glob.glob("../data/*.wav"):
        sound = parselmouth.Sound(wave_file)
        measurePitch(sound, 75, 500, "Hertz")

        # (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) =
        # file_list.append(wave_file) # make an ID list
        # mean_F0_list.append(meanF0) # make a mean F0 list
        # sd_F0_list.append(stdevF0) # make a sd F0 list
        # hnr_list.append(hnr)
        # localJitter_list.append(localJitter)
        # localabsoluteJitter_list.append(localabsoluteJitter)
        # rapJitter_list.append(rapJitter)
        # ppq5Jitter_list.append(ppq5Jitter)
        # ddpJitter_list.append(ddpJitter)
        # localShimmer_list.append(localShimmer)
        # localdbShimmer_list.append(localdbShimmer)
        # apq3Shimmer_list.append(apq3Shimmer)
        # aqpq5Shimmer_list.append(aqpq5Shimmer)
        # apq11Shimmer_list.append(apq11Shimmer)
        # ddaShimmer_list.append(ddaShimmer)

    df = pd.DataFrame(np.column_stack([file_list, mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, apq11Shimmer_list, ddaShimmer_list]),
                                   columns=['voiceID', 'meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter',
                                            'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer',
                                            'apq11Shimmer', 'ddaShimmer'])

    df.to_csv("../data/processed_results.csv", index=False)