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
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

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
    from insightface.app.common import Face as InsightFaceFace
except ImportError as exc:  # pragma: no cover - dependência externa
    raise ImportError(
        "O pacote 'insightface' é obrigatório. Instale com 'pip install insightface'."
    ) from exc


MODEL_DIR = pathlib.Path("models_insightface")
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_TARGET_FPS = 28.0

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


@dataclass
class AttributeTrack:
    """Armazena as informações suavizadas de uma pessoa reconhecida."""

    embedding: np.ndarray
    age: float
    gender_score: float
    last_seen: int


class AttributeSmoother:
    """Realiza suavização temporal das estimativas de idade e gênero."""

    def __init__(
        self,
        similarity_threshold: float = 0.32,
        max_unseen_frames: int = 60,
        age_momentum: float = 0.65,
        gender_momentum: float = 0.7,
        embedding_blend: float = 0.5,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_unseen_frames = max_unseen_frames
        self.age_momentum = age_momentum
        self.gender_momentum = gender_momentum
        self.embedding_blend = embedding_blend
        self._tracks: list[AttributeTrack] = []
        self._frame_index = 0

    def begin_frame(self) -> None:
        """Avança o contador de frames do suavizador."""

        self._frame_index += 1

    def _match_track(self, embedding: np.ndarray | None) -> AttributeTrack | None:
        """Localiza a trilha com maior similaridade para o embedding informado."""

        if embedding is None:
            return None
        embedding = np.asarray(embedding, dtype=np.float32)
        best_track: AttributeTrack | None = None
        best_sim = self.similarity_threshold
        for track in self._tracks:
            similarity = float(np.dot(track.embedding, embedding))
            if similarity > best_sim:
                best_sim = similarity
                best_track = track
        return best_track

    def update(self, embedding: np.ndarray | None, age: float, gender_idx: int) -> tuple[int, int]:
        """Atualiza as trilhas conhecidas e devolve valores suavizados."""

        if embedding is None:
            return int(round(age)), gender_idx

        embedding = np.asarray(embedding, dtype=np.float32)
        track = self._match_track(embedding)
        if track is None:
            track = AttributeTrack(
                embedding=embedding,
                age=float(age),
                gender_score=float(gender_idx),
                last_seen=self._frame_index,
            )
            self._tracks.append(track)
            return int(round(age)), gender_idx

        track.embedding = (
            track.embedding * (1.0 - self.embedding_blend) + embedding * self.embedding_blend
        )
        track.embedding /= max(np.linalg.norm(track.embedding), 1e-6)
        track.age = track.age * self.age_momentum + float(age) * (1.0 - self.age_momentum)
        track.gender_score = (
            track.gender_score * self.gender_momentum + float(gender_idx) * (1.0 - self.gender_momentum)
        )
        track.last_seen = self._frame_index
        smoothed_gender = int(round(track.gender_score))
        return int(round(track.age)), smoothed_gender

    def prune(self) -> None:
        """Remove trilhas que não são atualizadas há muito tempo."""

        self._tracks = [
            track
            for track in self._tracks
            if self._frame_index - track.last_seen <= self.max_unseen_frames
        ]


class PerformanceMonitor:
    """Mede latência e calcula FPS médio em uma janela deslizante."""

    def __init__(self, max_samples: int = 120) -> None:
        self._samples = deque(maxlen=max_samples)
        self._start_time: float | None = None

    def begin(self) -> None:
        """Marca o início de uma medição."""

        self._start_time = time.perf_counter()

    def end(self) -> float:
        """Conclui a medição e devolve a latência (em segundos)."""

        if self._start_time is None:
            return 0.0
        latency = time.perf_counter() - self._start_time
        self._samples.append(latency)
        self._start_time = None
        return latency

    @property
    def fps(self) -> float:
        """Retorna o FPS médio considerando as amostras recentes."""

        if not self._samples:
            return 0.0
        avg_latency = float(sum(self._samples) / len(self._samples))
        return 1.0 / avg_latency if avg_latency > 0.0 else 0.0

    @property
    def last_latency_ms(self) -> float:
        """Retorna a última latência registrada em milissegundos."""

        if not self._samples:
            return 0.0
        return float(self._samples[-1] * 1000.0)


class DynamicResizer:
    """Ajusta dinamicamente o tamanho de inferência para equilibrar FPS e qualidade."""

    def __init__(
        self,
        base_side: int = 960,
        min_side: int = 640,
        max_side: int = 1440,
        target_fps: float = DEFAULT_TARGET_FPS,
        step: int = 80,
    ) -> None:
        self._current_side = base_side
        self._min_side = min_side
        self._max_side = max_side
        self._target_fps = target_fps
        self._step = step
        self._last_scale = 1.0

    @property
    def last_scale(self) -> float:
        """Escala aplicada na última chamada de resize."""

        return self._last_scale

    def resize(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Redimensiona o frame respeitando a configuração dinâmica atual."""

        height, width = frame.shape[:2]
        longest_side = max(height, width)
        if longest_side <= self._current_side:
            self._last_scale = 1.0
            return frame, 1.0
        scale = self._current_side / float(longest_side)
        resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self._last_scale = scale
        return resized, scale

    def update(self, latency_seconds: float) -> None:
        """Atualiza a resolução alvo com base na latência observada."""

        if latency_seconds <= 0.0:
            return
        current_fps = 1.0 / latency_seconds
        if current_fps < self._target_fps * 0.9:
            self._current_side = max(self._min_side, self._current_side - self._step)
        elif current_fps > self._target_fps * 1.25 and self._last_scale == 1.0:
            self._current_side = min(self._max_side, self._current_side + self._step)
        elif current_fps > self._target_fps * 1.15:
            self._current_side = min(self._max_side, self._current_side + self._step // 2)


class CameraStream:
    """Captura frames da webcam em uma thread dedicada para reduzir latência."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture = capture
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._capture.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                self._latest_frame = frame

    def read(self) -> np.ndarray | None:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._capture.release()


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
        if "genderage" not in analyzer.models:
            raise ModelSetupError(
                "O modelo de idade/gênero não foi carregado corretamente."
            )
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


def resize_for_inference(frame: np.ndarray, max_side: int = 960) -> tuple[np.ndarray, float]:
    """Redimensiona o frame para acelerar a inferência mantendo a proporção."""

    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return frame, 1.0
    scale = max_side / float(longest_side)
    resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def clone_face(face: InsightFaceFace) -> InsightFaceFace:
    """Cria uma cópia independente de uma face detectada."""

    cloned = InsightFaceFace(face)
    for attr_name in ("bbox", "landmark", "landmark_2d_106", "landmark_3d_68", "kps"):
        value = getattr(face, attr_name, None)
        if value is not None:
            cloned[attr_name] = np.array(value, copy=True)
    if face.embedding is not None:
        cloned.embedding = np.array(face.embedding, copy=True)
    return cloned


def rescale_face_geometry(face: InsightFaceFace, scale: float) -> None:
    """Ajusta a geometria da face para corresponder ao frame original."""

    if scale == 1.0:
        return
    inv_scale = 1.0 / scale
    for attr_name in ("bbox", "landmark", "landmark_2d_106", "landmark_3d_68", "kps"):
        value = getattr(face, attr_name, None)
        if value is not None:
            face[attr_name] = np.asarray(value, dtype=np.float32) * inv_scale


def expand_bbox(box: Sequence[float], frame_shape: tuple[int, int, int], scale: float) -> np.ndarray:
    """Expande a caixa delimitadora respeitando os limites do frame."""

    x1, y1, x2, y2 = map(float, box)
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width / 2.0
    cy = y1 + height / 2.0
    half_w = (width * scale) / 2.0
    half_h = (height * scale) / 2.0
    new_x1 = max(0.0, cx - half_w)
    new_y1 = max(0.0, cy - half_h)
    new_x2 = min(float(frame_shape[1] - 1), cx + half_w)
    new_y2 = min(float(frame_shape[0] - 1), cy + half_h)
    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)


def estimate_age_gender(
    frame: np.ndarray,
    face: InsightFaceFace,
    attribute_model,
) -> tuple[float, int]:
    """Executa múltiplas inferências para obter idade e gênero mais robustos."""

    base_bbox = np.asarray(face.bbox, dtype=np.float32)
    scale_factors = (1.35, 1.55, 1.75)
    age_predictions: list[float] = []
    gender_predictions: list[int] = []
    for factor in scale_factors:
        jittered = clone_face(face)
        jittered.bbox = expand_bbox(base_bbox, frame.shape, factor)
        gender, age = attribute_model.get(frame, jittered)
        gender_predictions.append(int(gender))
        age_predictions.append(float(age))
    avg_age = float(np.mean(age_predictions)) if age_predictions else float(face.age or 0)
    avg_gender = int(round(float(np.mean(gender_predictions)))) if gender_predictions else int(face.gender or 0)
    return avg_age, avg_gender


def analyze_frame(
    frame: np.ndarray,
    analyzer: InsightFaceAnalysis,
    smoother: AttributeSmoother,
    resize_policy: DynamicResizer | None = None,
) -> tuple[np.ndarray, int, float]:
    """Processa um frame e retorna o frame anotado, número de faces e escala aplicada."""

    smoother.begin_frame()
    if resize_policy is None:
        processed_frame, scale = resize_for_inference(frame)
    else:
        processed_frame, scale = resize_policy.resize(frame)
    faces = analyzer.get(processed_frame)
    attribute_model = analyzer.models.get("genderage")
    if attribute_model is None:
        return frame, 0, scale

    cloned_faces = []
    for detected_face in faces:
        face = clone_face(detected_face)
        rescale_face_geometry(face, scale)
        cloned_faces.append(face)

    for face in cloned_faces:
        try:
            box = clip_bbox(face.bbox, frame.shape)
        except ValueError:
            continue
        x1, y1, x2, y2 = box
        face_roi = frame[y1:y2, x1:x2]

        age_estimate, gender_idx = estimate_age_gender(frame, face, attribute_model)
        age, gender_idx = smoother.update(face.normed_embedding, age_estimate, gender_idx)
        gender_label = GENDERS.get(gender_idx, "Desconhecido")
        age_label = f"{age} anos (aprox.)"
        skin_label = infer_skin_tone(face_roi)

        detection = Detection(box=box, confidence=float(face.det_score))
        attributes = FaceAttributes(age_label, gender_label, skin_label)
        annotate_frame(frame, detection, attributes)

    smoother.prune()
    return frame, len(cloned_faces), scale


def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """Inicializa a captura da webcam com algumas verificações de erro."""

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            "Não foi possível acessar a webcam. Verifique se ela está conectada e não está em uso por outro aplicativo."
        )
    return capture


def configure_capture(capture: cv2.VideoCapture) -> None:
    """Ajusta parâmetros do dispositivo para reduzir latência e estabilizar FPS."""

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_FPS, 30)
    buffersize_prop = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
    if buffersize_prop is not None:
        capture.set(buffersize_prop, 1)
    fourcc_prop = getattr(cv2, "CAP_PROP_FOURCC", None)
    if fourcc_prop is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        capture.set(fourcc_prop, fourcc)


def overlay_runtime_info(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    face_count: int,
    scale: float,
) -> None:
    """Exibe métricas de desempenho e escala de inferência no frame."""

    info_lines = [
        f"FPS: {fps:.1f}",
        f"Latência: {latency_ms:.1f} ms",
        f"Faces: {face_count}",
        f"Escala inferência: {scale * 100:.0f}%",
    ]
    for idx, text in enumerate(info_lines):
        y = 20 + idx * 18
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def run(camera_index: int = 0) -> None:
    """Loop principal do aplicativo."""

    analyzer = create_face_analyzer()
    smoother = AttributeSmoother()

    try:
        capture = open_camera(camera_index)
    except RuntimeError as err:
        print(err)
        return

    configure_capture(capture)
    cv2.setUseOptimized(True)

    stream = CameraStream(capture)
    stream.start()
    monitor = PerformanceMonitor()
    resizer = DynamicResizer()

    print('Pressione "q" na janela da webcam para sair.')

    try:
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.005)
                continue

            monitor.begin()
            try:
                analyzed_frame, face_count, scale = analyze_frame(
                    frame, analyzer, smoother, resizer
                )
            finally:
                latency_seconds = monitor.end()

            resizer.update(latency_seconds)
            overlay_runtime_info(
                analyzed_frame,
                monitor.fps,
                monitor.last_latency_ms,
                face_count,
                scale,
            )
            cv2.imshow("Análise facial", analyzed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
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
