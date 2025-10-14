"""Aplicativo de análise facial em tempo real.

Este módulo abre a webcam local, detecta faces e estima idade, gênero e tom de pele
para cada pessoa enquadrada. Modelos pré-treinados da OpenCV são baixados
automaticamente caso não estejam disponíveis localmente.

Execute com:
    python main.py

Pressione a tecla "q" na janela da webcam para encerrar o programa.
"""
from __future__ import annotations

import pathlib
import sys
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np


MODEL_SPECS = {
    "face": (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    ),
    "age": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_models/age_deploy.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_models/age_net.caffemodel",
    ),
    "gender": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_models/gender_deploy.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_models/gender_net.caffemodel",
    ),
}

AGE_BUCKETS = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]
GENDERS = ["Feminino", "Masculino"]

SKIN_TONE_LABELS = {
    "light": "Tom de pele: claro",
    "medium": "Tom de pele: médio",
    "dark": "Tom de pele: escuro",
}

MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(exist_ok=True)


class ModelDownloadError(RuntimeError):
    """Exceção lançada quando o download de um modelo falhar."""


@dataclass
class Detection:
    """Representa uma detecção de face."""

    box: Tuple[int, int, int, int]
    confidence: float


@dataclass
class FaceAnalysis:
    """Resultado da análise facial."""

    age_label: str
    gender_label: str
    skin_tone_label: str


def download_models() -> dict[str, Tuple[pathlib.Path, pathlib.Path]]:
    """Garante que todos os arquivos de modelo estejam presentes localmente."""

    model_files: dict[str, Tuple[pathlib.Path, pathlib.Path]] = {}
    for name, (proto_url, weights_url) in MODEL_SPECS.items():
        proto_path = MODEL_DIR / f"{name}.prototxt"
        weights_suffix = pathlib.Path(weights_url).suffix
        weights_path = MODEL_DIR / f"{name}{weights_suffix}"

        for url, path in ((proto_url, proto_path), (weights_url, weights_path)):
            if path.exists():
                continue
            try:
                print(f"Baixando {name} de {url}...", flush=True)
                urllib.request.urlretrieve(url, path)
            except OSError as exc:  # inclui URLError
                raise ModelDownloadError(
                    f"Falha ao baixar o arquivo de modelo '{path.name}' a partir de {url}"
                ) from exc
        model_files[name] = (proto_path, weights_path)
    return model_files


def load_dnn_models(model_files: dict[str, Tuple[pathlib.Path, pathlib.Path]]):
    """Carrega os modelos DNN da OpenCV."""

    face_net = cv2.dnn.readNetFromCaffe(*map(str, model_files["face"]))
    age_net = cv2.dnn.readNetFromCaffe(*map(str, model_files["age"]))
    gender_net = cv2.dnn.readNetFromCaffe(*map(str, model_files["gender"]))
    return face_net, age_net, gender_net


def detect_faces(face_net: cv2.dnn_Net, frame: np.ndarray, threshold: float = 0.5) -> Iterable[Detection]:
    """Executa a detecção de faces em um frame BGR."""

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < threshold:
            continue
        x1 = max(int(detections[0, 0, i, 3] * w), 0)
        y1 = max(int(detections[0, 0, i, 4] * h), 0)
        x2 = min(int(detections[0, 0, i, 5] * w), w - 1)
        y2 = min(int(detections[0, 0, i, 6] * h), h - 1)
        yield Detection(box=(x1, y1, x2, y2), confidence=confidence)


def infer_age_gender(
    age_net: cv2.dnn_Net, gender_net: cv2.dnn_Net, face_roi: np.ndarray
) -> Tuple[str, str]:
    """Realiza a inferência de idade e gênero para uma imagem de face (BGR)."""

    blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()[0]
    gender_idx = int(np.argmax(gender_preds))
    gender_label = GENDERS[gender_idx]

    age_net.setInput(blob)
    age_preds = age_net.forward()[0]
    age_idx = int(np.argmax(age_preds))
    age_label = AGE_BUCKETS[age_idx]

    return age_label, gender_label


def infer_skin_tone(face_roi: np.ndarray) -> str:
    """Classifica o tom de pele da face com base na luminosidade média."""

    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]
    mean_v = float(np.mean(v_channel))
    mean_s = float(np.mean(s_channel))

    # Ajuste simples: valores muito saturados e escuros são considerados escuros.
    if mean_v < 80 or (mean_v < 100 and mean_s > 80):
        return SKIN_TONE_LABELS["dark"]
    if mean_v < 150:
        return SKIN_TONE_LABELS["medium"]
    return SKIN_TONE_LABELS["light"]


def annotate_frame(frame: np.ndarray, detection: Detection, analysis: FaceAnalysis) -> None:
    """Desenha a caixa delimitadora e informações no frame."""

    x1, y1, x2, y2 = detection.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"Idade: {analysis.age_label} | Sexo: {analysis.gender_label}"
    skin_label = analysis.skin_tone_label

    for idx, text in enumerate((label, skin_label)):
        y = y1 - 10 - idx * 18
        if y < 10:
            y = y1 + 20 + idx * 18
        cv2.putText(
            frame,
            text,
            (x1, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def analyze_frame(
    frame: np.ndarray, face_net: cv2.dnn_Net, age_net: cv2.dnn_Net, gender_net: cv2.dnn_Net
) -> np.ndarray:
    """Processa um frame e retorna o frame anotado."""

    for detection in detect_faces(face_net, frame):
        x1, y1, x2, y2 = detection.box
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue
        age_label, gender_label = infer_age_gender(age_net, gender_net, face_roi)
        skin_label = infer_skin_tone(face_roi)
        analysis = FaceAnalysis(age_label, gender_label, skin_label)
        annotate_frame(frame, detection, analysis)
    return frame


def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """Inicializa a captura da webcam com algumas verificações de erro."""

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            "Não foi possível acessar a webcam. Verifique se ela está conectada e não está em uso por outro aplicativo."
        )
    return capture


def run(camera_index: int = 0) -> None:
    """Loop principal do aplicativo."""

    model_files = download_models()
    face_net, age_net, gender_net = load_dnn_models(model_files)

    try:
        capture = open_camera(camera_index)
    except RuntimeError as err:
        print(err)
        return

    print('Pressione "q" na janela da webcam para sair.')
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Falha ao capturar frame da webcam. Encerrando...")
            break

        analyzed_frame = analyze_frame(frame, face_net, age_net, gender_net)
        cv2.imshow("Análise facial", analyzed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def main(argv: Iterable[str] | None = None) -> int:
    """Ponto de entrada do script."""

    if argv is None:
        argv = sys.argv[1:]

    camera_index = 0
    if argv:
        try:
            camera_index = int(argv[0])
        except ValueError:
            print("O índice da câmera deve ser um número inteiro.")
            return 1

    try:
        run(camera_index)
    except ModelDownloadError as exc:
        print(exc)
        return 1
    except cv2.error as exc:
        print(f"Erro da OpenCV: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
