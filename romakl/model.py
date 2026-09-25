import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"  # Lalamo otherwise sets .95 on import.

from lalamo import ChatCodec, ChatCodecConfig, Message
from lalamo.models.language_model import GenerationConfig
from tokenizers import Tokenizer

from .rollouts import Rollout


class ModelEngine(StrEnum):
    LALAMO = "lalamo"
    LCPP = "lcpp"
    EXL = "exl"


@dataclass(frozen=True)
class Model(ABC):
    codec: ChatCodec
    path: Path | str

    @staticmethod
    def init(specification: str, codec_config: ChatCodecConfig | None = None) -> "Model":
        engine_name, model_path = specification.split(":", 1)
        engine = ModelEngine(engine_name)
        path = Path(model_path).expanduser()
        if engine is ModelEngine.LALAMO:
            from .vendors.lalamo import LalamoModel

            return LalamoModel.load(path)

        assert codec_config is not None, "non-Lalamo models require the reference chat codec config"
        if engine is ModelEngine.LCPP:
            from .vendors.lcpp import GGUFModel

            tokenizer_path = path.with_name("tokenizer.json")
            return GGUFModel(codec_config.init(Tokenizer.from_file(str(tokenizer_path))), path)

        from .vendors.exl import ExLlamaModel

        tokenizer_path = path / "tokenizer.json"
        return ExLlamaModel(codec_config.init(Tokenizer.from_file(str(tokenizer_path))), path)

    def tokenize(self, conversations: list[list[Message]]) -> list[list[int]]:
        return [self.codec.encode_request(conversation) for conversation in conversations]

    def detokenize(self, tokens: list[int]) -> str:
        return self.codec.decode_tokens(tokens)

    @abstractmethod
    def generate(
        self, prompts: list[list[int]], generation_config: GenerationConfig, *, seed: int, token_limit: int
    ) -> list[Rollout]: ...

    @abstractmethod
    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]: ...
