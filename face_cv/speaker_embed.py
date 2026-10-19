"""ECAPA embedding of a live clip on the care PC. Enrolled vectors are in Qdrant.

Torch and SpeechBrain are not part of the face-worker install. They live in
``requirements-voice.txt`` because TensorFlow 2.13 cannot share a casual
upgrade with them. Face recognition still runs if this import fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SPEECHBRAIN_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_SAMPLE_RATE = 16_000
_PRETRAINED_DIR = Path(__file__).resolve().parents[1] / "pretrained_models" / "spkrec-ecapa-voxceleb"

_encoder = None


def embed_numpy(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("empty audio")
    import torch

    waveform = torch.from_numpy(_resample(audio, sample_rate))
    return _embed_waveform(waveform)


def embed_file(path: str | Path) -> np.ndarray:
    import soundfile as sf

    data, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return embed_numpy(audio, int(sample_rate))


def _resample(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE:
        return samples
    from scipy import signal as scipy_signal

    n_out = int(round(len(samples) * TARGET_SAMPLE_RATE / float(sample_rate)))
    if n_out <= 0:
        return samples
    return scipy_signal.resample(samples, n_out).astype(np.float32)


def _embed_waveform(waveform) -> np.ndarray:
    import torch

    if waveform.dim() == 1:
        wav = waveform.unsqueeze(0)
    else:
        wav = waveform
    encoder = _get_encoder()
    with torch.no_grad():
        emb = encoder.encode_batch(wav)
        emb = emb.squeeze()
        if emb.dim() > 1:
            emb = emb.mean(dim=0)
        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=0)
    return emb.cpu().numpy().astype(np.float64)


def _get_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    import torch

    _patch_torch_amp(torch)
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as e:  # noqa: BLE001 — missing wheel or WinError 127
        raise ImportError(
            "SpeechBrain is not installed. From the face-worker repo: "
            "pip install -r requirements-voice.txt. "
            f"Original error: {e}"
        ) from e
    _PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    _encoder = EncoderClassifier.from_hparams(
        source=SPEECHBRAIN_SOURCE,
        savedir=str(_PRETRAINED_DIR),
        run_opts={"device": "cpu"},
    )
    return _encoder


def _patch_torch_amp(torch) -> None:
    if hasattr(torch.amp, "custom_fwd"):
        return

    def custom_fwd(fwd=None, *, device_type=None, cast_inputs=None):  # noqa: ARG001
        return torch.cuda.amp.custom_fwd(fwd, cast_inputs=cast_inputs)

    def custom_bwd(bwd=None, *, device_type=None):  # noqa: ARG001
        return torch.cuda.amp.custom_bwd(bwd)

    torch.amp.custom_fwd = custom_fwd
    torch.amp.custom_bwd = custom_bwd
