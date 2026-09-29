import json
import os
import re
from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # Share the GPU with non-lalamo students.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/romakl-cache")  # noqa: S108

from lalamo import ChatCodec, ChatCodecConfig
from lalamo.models.language_model import GenerationConfig
from tokenizers import Tokenizer

from .dataset import Conversation
from .rollouts import Rollout


class ModelEngine(StrEnum):
    LALAMO = "lalamo"
    LCPP = "lcpp"
    EXL = "exl"


class StudentModel(Protocol):
    @abstractmethod
    def tokenize(self, conversations: list[Conversation]) -> list[list[int]]: ...

    @abstractmethod
    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]: ...

    @abstractmethod
    def generator_bytes(self) -> int: ...


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

    def generator_bytes(self) -> int:
        # Vision/audio towers and MTP heads, as named across the architectures lalamo supports.
        auxiliary = re.compile(
            r"(^|\.)(mtp|visual|vision_\w+|multi_modal_projector|embed_vision|embed_audio|audio_\w+)(\.|$)"
        )
        total = 0
        for path in Path(self.path).glob("*.safetensors"):
            with path.open("rb") as file:
                header = json.loads(file.read(int.from_bytes(file.read(8), "little")))
            total += sum(
                info["data_offsets"][1] - info["data_offsets"][0]
                for name, info in header.items()
                if name != "__metadata__" and not auxiliary.search(name)
            )
        return total

    def tokenize(self, conversations: list[Conversation]) -> list[list[int]]:
        return [
            self.codec.encode_request(
                conversation.messages, tools=conversation.tools, reasoning_effort=conversation.reasoning_effort
            )
            for conversation in conversations
        ]


# Vendors subclass Model, so they can only be imported once it is defined.
from .vendors import exl, lalamo, lcpp  # noqa: E402
