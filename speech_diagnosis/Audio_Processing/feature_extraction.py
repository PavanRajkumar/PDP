import parselmouth
from parselmouth.praat import call
import glob
import re
import numpy as np

class Feature_extraction:

    def __init__(self, sound_location):

        self.voiceID= self.get_sound(sound_location)

        #Pitch
        self.pitch_mean  = None
        self.pitch_median = None
        self.pitch_std_dev = None
        self.pitch_minimum = None
        self.pitch_maximum = None

        #Harmonicity
        # self.harmonicity_mean = None
        self.AC = None
        self.HTN  =None
        self.NTH  = None

        #Jitter
        self.jitter_local  =None
        self.jitter_absolute_local  = None
        self.jitter_rap  = None
        self.jitter_ppq5  =None
        self.jitter_ddp  = None

        #Shimmer
        self.shimmer_local  = None
        self.shimmer_local_db  =None
        self.shimmer_apq3   = None
        self.shimmer_aqpq5   = None
        self.shimmer_apq11  = None
        self.shimmer_dda  = None

        #Pulse
        self.number_of_pulses = None
        self.number_of_periods = None
        self.mean_period = None
        self.std_dev_period = None

        #Voicing
        self.fraction_unvoiced_frames = None
        self.number_voice_breaks = None
        self.degree_voice_breaks = None

        #Overall Report
        self.vocal_report = None
        self.all_vocal_parameters = None

    def get_sound(self, sound_location):

        voiceId = parselmouth.Sound(sound_location)

        return voiceId

    def get_pitch_parameters(self, Pitch, unit):

        self.pitch_mean = call(Pitch, "Get mean", 0, 0, unit)  # get mean pitch
        self.pitch_median = float(re.findall("Median pitch: ([0-9]*\.[0-9]*)",
                                             self.vocal_report)[0])
        self.pitch_std_dev = call(Pitch, "Get standard deviation", 0, 0, unit)  # get standard deviation
        self.pitch_minimum = float(re.findall("Minimum pitch: ([0-9]*\.[0-9]*)",
                                              self.vocal_report)[0])
        self.pitch_maximum = float(re.findall("Maximum pitch: ([0-9]*\.[0-9]*)",
                                              self.vocal_report)[0])

        return None

    def get_harmonicity_parameters(self, sound):

        Harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        # self.harmonicity_mean = call(Harmonicity, "Get mean", 0, 0)

        self.AC = float(re.findall("Mean autocorrelation: ([0-9]*\.[0-9]*)",
                                                              self.vocal_report)[0])
        self.NTH = float(re.findall("Mean noise-to-harmonics ratio: ([0-9]*\.[0-9]*)",
                                                              self.vocal_report)[0])
        self.HTN = float(re.findall("Mean harmonics-to-noise ratio: ([0-9]*\.[0-9]*)",
                                                              self.vocal_report)[0])

        return Harmonicity


    def get_jitter_parameters(self, sound, pointProcess):

        self.jitter_local = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        self.jitter_absolute_local = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        self.jitter_rap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        self.jitter_ppq5 = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        self.jitter_ddp = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

        return None

    def get_shimmer_parameters(self, sound, pointProcess):

        self.shimmer_local = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        self.shimmer_local_db = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        self.shimmer_apq3 = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        self.shimmer_aqpq5 = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        self.shimmer_apq11 = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        self.shimmer_dda = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        return None

    def get_pulse_parameters(self):

        self.number_of_pulses = float(re.findall("Number of pulses: ([0-9]*)", self.vocal_report)[0])
        self.number_of_periods = float(re.findall("Number of periods: ([0-9]*)", self.vocal_report)[0])
        self.mean_period = float(re.findall("Mean period: ([0-9]*\.[0-9]*[Ee]-[0-9]*)", self.vocal_report)[0])
        self.std_dev_period = float(re.findall("Standard deviation of period: ([0-9]*\.[0-9]*[Ee]-[0-9]*)",
                                               self.vocal_report)[0])

    def get_voicing_parameters(self):

        self.fraction_unvoiced_frames = float(re.findall("Fraction of locally unvoiced frames: ([0-9]*\.[0-9]*)",
                                                         self.vocal_report)[0])
        self.fraction_unvoiced_frames = self.fraction_unvoiced_frames/100.0

        self.number_voice_breaks = float(re.findall("Number of voice breaks: ([0-9]*)", self.vocal_report)[0])

        self.degree_voice_breaks = float(re.findall("Degree of voice breaks: ([0-9]*\.[0-9]*)", self.vocal_report)[0])
        self.degree_voice_breaks = self.degree_voice_breaks/100.0


    def get_all_features(self, f0min, f0max, unit):

        try:

            sound = parselmouth.Sound(self.voiceID) # read the sound
            pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
            Pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
            #Vocal Report
            self.vocal_report = parselmouth.praat.call([sound, Pitch, pointProcess], "Voice report", 0, 0, 75, 600, 1.3, 1.6, 0.03, 0.45)

            #Pitch
            self.get_pitch_parameters(Pitch=Pitch, unit=unit)

            #Harmonicity
            Harmonicity = self.get_harmonicity_parameters(sound)

            #Jitter
            self.get_jitter_parameters(sound=sound, pointProcess=pointProcess)

            #Shimmer

            self.get_shimmer_parameters(sound=sound, pointProcess=pointProcess)

            #Pulse
            self.get_pulse_parameters()

            #Voicing
            self.get_voicing_parameters()

        except Exception as e:
            pass

        self.all_vocal_parameters = self.__dict__

        return self.all_vocal_parameters

    def get_features_as_numpy(self):

        if self.all_vocal_parameters is None:
            self.get_all_features(75, 500, "Hertz")

        columns = ["jitter_local", "jitter_absolute_local", "jitter_rap", "jitter_ppq5",
                   "jitter_ddp", "shimmer_local", "shimmer_local_db", "shimmer_apq3", "shimmer_aqpq5",
                   "shimmer_apq11", "shimmer_dda","AC", "NTH", "HTN", "pitch_median", "pitch_mean",
                   "pitch_std_dev", "pitch_minimum", "pitch_maximum","number_of_pulses", "number_of_periods",
                   "mean_period", "std_dev_period", "fraction_unvoiced_frames", "number_voice_breaks",
                   "degree_voice_breaks"]

        data_instance = []

        for cols in columns:
            data_instance.append(self.all_vocal_parameters[cols])

        data_instance = np.array(data_instance).reshape(1, 26)

        return data_instance

    def get_voice_report(self):
        return self.vocal_report



if __name__ == "__main__":


    for wave_file in glob.glob("../data/*.wav"):

        features = Feature_extraction(wave_file)
        result = features.get_all_features(75, 500, "Hertz")
        # result = features.__dict__
        print(features.get_features_as_numpy())

        break