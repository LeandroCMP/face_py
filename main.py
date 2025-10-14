"""Aplicativo de análise facial em tempo real.

Este módulo abre a webcam local, detecta faces e estima idade, gênero e tom de pele
para cada pessoa enquadrada utilizando os modelos mais recentes do pacote InsightFace.
Os artefatos de modelo são baixados automaticamente a partir dos releases oficiais
quando executados pela primeira vez.

Pré-requisitos:
    pip install opencv-python insightface onnxruntime

Execute com:
    python main.py [indice_camera]

Pressione a tecla "q" na janela da webcam para encerrar o programa.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - dependência externa
    raise ImportError(
        "O pacote 'onnxruntime' é obrigatório. Instale com 'pip install onnxruntime'."
    ) from exc

try:
    from insightface.app import FaceAnalysis as InsightFaceAnalysis
except ImportError as exc:  # pragma: no cover - dependência externa
    raise ImportError(
        "O pacote 'insightface' é obrigatório. Instale com 'pip install insightface'."
    ) from exc


MODEL_DIR = pathlib.Path("models_insightface")
MODEL_DIR.mkdir(exist_ok=True)

GENDERS = {0: "Feminino", 1: "Masculino"}
SKIN_TONE_LABELS = {
    "light": "Tom de pele: claro",
    "medium": "Tom de pele: médio",
    "dark": "Tom de pele: escuro",
}


class ModelSetupError(RuntimeError):
    """Exceção lançada quando a preparação dos modelos falha."""


@dataclass
class Detection:
    """Representa uma detecção de face."""

    box: tuple[int, int, int, int]
    confidence: float


@dataclass
class FaceAttributes:
    """Resultado da análise facial."""

    age_label: str
    gender_label: str
    skin_tone_label: str


def get_onnx_providers() -> list[str]:
    """Seleciona os providers disponíveis do ONNX Runtime."""

    preferred = [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    available = ort.get_available_providers()
    providers = [provider for provider in preferred if provider in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    return providers


def create_face_analyzer(model_dir: pathlib.Path = MODEL_DIR) -> InsightFaceAnalysis:
    """Inicializa o analisador do InsightFace."""

    providers = get_onnx_providers()
    try:
        analyzer = InsightFaceAnalysis(name="buffalo_l", root=str(model_dir), providers=providers)
        analyzer.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
    except Exception as exc:  # pragma: no cover - exceções vindas do backend
        raise ModelSetupError(
            "Falha ao preparar os modelos do InsightFace. Verifique a conexão com a internet "
            "para baixar os artefatos oficiais ou tente reinstalar as dependências."
        ) from exc
    return analyzer


def clip_bbox(box: Iterable[float], frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Ajusta a caixa delimitadora para ficar dentro dos limites da imagem."""

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = (int(round(coord)) for coord in box)
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w - 1)
    y2 = min(y2, h - 1)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Caixa delimitadora inválida gerada pelo modelo.")
    return x1, y1, x2, y2


def infer_skin_tone(face_roi: np.ndarray) -> str:
    """Classifica o tom de pele da face com base na luminosidade média em LAB."""

    if face_roi.size == 0:
        return SKIN_TONE_LABELS["medium"]

    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    mean_l = float(np.mean(l_channel))

    if mean_l < 80:
        return SKIN_TONE_LABELS["dark"]
    if mean_l < 160:
        return SKIN_TONE_LABELS["medium"]
    return SKIN_TONE_LABELS["light"]


def annotate_frame(frame: np.ndarray, detection: Detection, attributes: FaceAttributes) -> None:
    """Desenha a caixa delimitadora e informações no frame."""

    x1, y1, x2, y2 = detection.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    info_lines = [
        f"Idade: {attributes.age_label} | Sexo: {attributes.gender_label}",
        attributes.skin_tone_label,
        f"Confiança: {detection.confidence:.2f}",
    ]
    for idx, text in enumerate(info_lines):
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


def analyze_frame(frame: np.ndarray, analyzer: InsightFaceAnalysis) -> np.ndarray:
    """Processa um frame e retorna o frame anotado."""

    faces = analyzer.get(frame)
    for face in faces:
        try:
            box = clip_bbox(face.bbox, frame.shape)
        except ValueError:
            continue
        x1, y1, x2, y2 = box
        face_roi = frame[y1:y2, x1:x2]

        age = int(round(face.age))
        gender_idx = int(round(face.gender))
        gender_label = GENDERS.get(gender_idx, "Desconhecido")
        age_label = f"{age} anos (aprox.)"
        skin_label = infer_skin_tone(face_roi)

        detection = Detection(box=box, confidence=float(face.det_score))
        attributes = FaceAttributes(age_label, gender_label, skin_label)
        annotate_frame(frame, detection, attributes)
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

    analyzer = create_face_analyzer()

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

        analyzed_frame = analyze_frame(frame, analyzer)
        cv2.imshow("Análise facial", analyzed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def parse_camera_index(argv: Iterable[str]) -> int:
    """Extrai o índice da câmera a partir dos argumentos de linha de comando."""

    values = list(argv)
    if not values:
        return 0
    try:
        return int(values[0])
    except ValueError as exc:
        raise ValueError("O índice da câmera deve ser um número inteiro.") from exc


def main(argv: Iterable[str] | None = None) -> int:
    """Ponto de entrada do script."""

    if argv is None:
        argv = sys.argv[1:]

    try:
        camera_index = parse_camera_index(argv)
    except ValueError as exc:
        print(exc)
        return 1

    try:
        run(camera_index)
    except ModelSetupError as exc:
        print(exc)
        return 1
    except cv2.error as exc:  # pragma: no cover - erros de backend do OpenCV
        print(f"Erro da OpenCV: {exc}")
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interrupção manual
        print("\nExecução interrompida pelo usuário.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
