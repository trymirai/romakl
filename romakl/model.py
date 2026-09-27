from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
from lalamo import ChatCodec, ChatCodecConfig, Message
from lalamo.models.language_model import GenerationConfig
from tokenizers import Tokenizer

from .rollouts import Rollout


class ModelEngine(StrEnum):
    LALAMO = "lalamo"
    LCPP = "lcpp"
    EXL = "exl"


class StudentModel(Protocol):
    @abstractmethod
    def tokenize(self, conversations: list[list[Message]]) -> list[list[int]]: ...

    @abstractmethod
    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]: ...


class TeacherModel(Protocol):
    @abstractmethod
    def generate(
        self, prompts: list[list[int]], generation_config: GenerationConfig, *, seed: int, token_limit: int
    ) -> list[Rollout]: ...


@dataclass(frozen=True)
class Model:
    codec: ChatCodec
    path: Path | str

    @staticmethod
    def init(specification: str, codec_config: ChatCodecConfig) -> StudentModel:
        engine_name, model_path = specification.split(":", 1)
        path = Path(model_path).expanduser()
        match ModelEngine(engine_name):
            case ModelEngine.LALAMO:
                return lalamo.LalamoModel.load(path)
            case ModelEngine.LCPP:
                return lcpp.GGUFModel(
                    codec_config.init(Tokenizer.from_file(str(path.with_name("tokenizer.json")))), path
                )
            case ModelEngine.EXL:
                return exl.ExLlamaModel(codec_config.init(Tokenizer.from_file(str(path / "tokenizer.json"))), path)

    def tokenize(self, conversations: list[list[Message]]) -> list[list[int]]:
        return [self.codec.encode_request(conversation) for conversation in conversations]


# Vendors subclass Model, so they can only be imported once it is defined.
from .vendors import exl, lalamo, lcpp  # noqa: E402
