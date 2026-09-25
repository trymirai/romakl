import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"  # Lalamo otherwise sets .95 on import.

from lalamo import ChatCodec

from .rollouts import Rollout


@dataclass(frozen=True)
class GenerationParams:
    seed: int = 0
    token_limit: int = 512

    temperature: float = 1.0
    top_k: int = 20
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass(frozen=True)
class Model(ABC):
    codec: ChatCodec
    path: Path | str

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        return [self.codec.encode_text(text) for text in texts]

    def detokenize(self, tokens: list[int]) -> str:
        return self.codec.decode_tokens(tokens)

    @abstractmethod
    def generate(self, prompts: list[list[int]], generation_params: GenerationParams) -> list[Rollout]: ...

    @abstractmethod
    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]: ...
