
import pickle

from utils import extract_feature


def main() -> None:
    # cargar el modelo guardado (después del entrenamiento)
    model = pickle.load(open("result/mlp_classifier-79.model", "rb"))
    filename = "test.wav"

    # extraer características y remodelarlas
    features = extract_feature(
        filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predecir
    resultado = model.predict(features)[0]
    # muestra el resultado!
    print("resultado:", resultado)

    return resultado


if __name__ == "__main__":
    main()
