import re
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from llama_cpp import Llama

from ..model import Model, StudentModel  # noqa: TID252
from ..rollouts import Rollout  # noqa: TID252


@dataclass(frozen=True)
class GGUFModel(Model, StudentModel):
    def generator_bytes(self) -> int:
        path = Path(self.path)
        # A split GGUF (name-00001-of-0000N.gguf) lists each shard's tensors only in that shard.
        shards = sorted(path.parent.glob(re.sub(r"-00001-of-(\d+)\.gguf$", r"-*-of-\1.gguf", path.name)))
        readers = [GGUFReader(shard) for shard in shards]
        architecture = readers[0].fields["general.architecture"].contents()
        block_count = readers[0].fields[f"{architecture}.block_count"].contents()
        nextn_field = readers[0].fields.get(f"{architecture}.nextn_predict_layers")
        mtp_count = nextn_field.contents() if nextn_field else 0
        # llama.cpp stores MTP heads as the last `nextn_predict_layers` blocks.
        mtp_prefixes = tuple(f"blk.{index}." for index in range(block_count - mtp_count, block_count))
        return sum(
            tensor.n_bytes
            for reader in readers
            for tensor in reader.tensors
            if not tensor.name.startswith(mtp_prefixes)
        )

    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]:
        result = []
        context = max(len(rollout.tokens) - 1 for rollout in rollouts)
        with closing(
            Llama(
                str(self.path),
                n_ctx=context,
                n_batch=context,
                n_ubatch=2048,
                n_gpu_layers=-1,
                logits_all=True,
                flash_attn=True,
                verbose=False,
            )
        ) as model:
            for rollout in rollouts:
                output_tokens = sum(rollout.mask)
                model.reset()
                # eval computes logits for every token it is given under logits_all, so the prompt runs without it.
                model._logits_all = False  # noqa: SLF001
                model.eval(rollout.tokens[: -output_tokens - 1])
                model._logits_all = True  # noqa: SLF001
                model.eval(rollout.tokens[-output_tokens - 1 : -1])
                result.append(model.scores[model.n_tokens - output_tokens : model.n_tokens].astype(np.float16))
        return result
