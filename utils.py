import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split

# todas las emociones en el conjunto de datos de RAVDESS
EmocionesRAVDESS = {
    "01": "Positivo",
    "02": "Negativo",
    "03": "feliz",
    "04": "triste",
    "05": "enojado",
    "06": "asustado",
    "07": "disgustado",
    "08": "sorprendido"
}
# solo permitimos estas emociones
AVAILABLE_EMOTIONS = {

    "Positivo",
    "Negativo",
}


def extract_feature(file_name, **kwargs):

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(
                y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result

# data/Actor_*/*.wav

def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("data/Actor_*/*.wav"):
        # obtener el nombre base del archivo de audio
        basename = os.path.basename(file)
        # obtener la etiqueta de emoción
        emocion = EmocionesRAVDESS[basename.split("-")[2]]
        # permitimos solo EMOCIONES DISPONIBLES que establezcamos
        if emocion not in AVAILABLE_EMOTIONS:
            continue
        # extraer características del habla
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # agregar a los datos
        X.append(features)
        y.append(emocion)
    # dividir los datos para entrenamiento y prueba y devolverlos
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)


if __name__ == "__main__":
    load_data()
