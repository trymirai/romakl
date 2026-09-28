# ruff: noqa: I001
from ..model import Model, StudentModel, TeacherModel  # noqa: TID252
from ..rollouts import Rollout  # noqa: TID252
import math
from dataclasses import dataclass, replace
from itertools import batched, groupby
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from lalamo import LanguageModel
from lalamo.models.language_model import GenerationConfig
from lalamo.module import Keychain
from lalamo.modules import DecoderForwardPassConfig
from lalamo.modules.token_mixer import AttentionImplementation
from lalamo.utils.sharding import ShardingConfig
from loguru import logger


@dataclass(frozen=True)
class LalamoModel(Model, TeacherModel, StudentModel):
    model: LanguageModel

    @classmethod
    def load(cls, path: Path | str) -> "LalamoModel":
        model = LanguageModel.load(path, ShardingConfig.replicated(), dtype=jnp.bfloat16)
        return cls(model.token_codec, path, model)

    def generate(
        self, prompts: list[list[int]], generation_config: GenerationConfig, *, seed: int, token_limit: int
    ) -> list[Rollout]:
        # Each distinct (batch, length) shape recompiles generate_tokens, so prompts are padded to power-of-two
        # buckets split into equal batches. Per-prompt keys keep sampling independent of the batching.
        keychain = Keychain.init(seed, shape=(len(prompts),), sharding_config=self.model.sharding_config)
        rollouts: dict[int, Rollout] = {}
        by_length = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
        for length, bucket_indices in groupby(
            by_length, key=lambda index: max(4096, 1 << (len(prompts[index]) - 1).bit_length())
        ):
            bucket = list(bucket_indices)
            batch_size = math.ceil(len(bucket) / math.ceil(len(bucket) / max(1, 131_072 // length)))
            for batch in batched(bucket, batch_size):
                logger.debug("Generating {} prompts padded to {} tokens", len(batch), length)
                filled = [*batch, *[batch[0]] * (batch_size - len(batch))]
                generated = self.model.generate_tokens(
                    jnp.asarray(
                        [prompts[index] + [0] * (length - len(prompts[index])) for index in filled], dtype=jnp.int32
                    ),
                    generation_config,
                    prompt_lengths_without_padding=jnp.asarray(
                        [len(prompts[index]) for index in filled], dtype=jnp.int32
                    ),
                    max_output_length=token_limit,
                    keychain=replace(keychain, vmapped_keys=keychain.vmapped_keys[jnp.asarray(filled)]),
                ).token_ids.tolist()
                for index, generated_tokens in zip(batch, generated[: len(batch)], strict=True):
                    tokens = self.model.trim_at_eos(generated_tokens)
                    rollouts[index] = Rollout(
                        [*prompts[index], *tokens], [0] * len(prompts[index]) + [1] * len(tokens)
                    )
        return [rollouts[index] for index in range(len(prompts))]

    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]:
        # Lalamo's Pallas attention only supports one query token; scoring needs every logit.
        config = DecoderForwardPassConfig.for_inference()
        transformer_config = config.transformer_forward_pass_config
        mixer_config = replace(
            transformer_config.mixer_forward_pass_config,
            attention_implementation=AttentionImplementation.XLA,
        )
        config = replace(
            config,
            transformer_forward_pass_config=replace(transformer_config, mixer_forward_pass_config=mixer_config),
        )
        chunk_size = 2048
        # A single state capacity keeps the jitted decoder to one compilation across all rollouts.
        capacity = math.ceil(max(len(rollout.tokens) for rollout in rollouts) / chunk_size) * chunk_size + 1
        keychain = Keychain.init(0, sharding_config=self.model.sharding_config)
        result = []
        for rollout in rollouts:
            logger.debug("Computing logits for {} tokens", len(rollout.tokens))
            length = len(rollout.tokens) - 1
            output_start = length - sum(rollout.mask)
            padded_length = math.ceil(length / chunk_size) * chunk_size
            token_ids = jnp.asarray([rollout.tokens[:-1] + [0] * (padded_length - length)], dtype=jnp.int32)
            state = self.model.decoder.init_static_state(
                1, capacity, config.embedding_forward_pass_config.activation_dtype
            )
            chunks = []
            for chunk_start in range(0, padded_length, chunk_size):
                decoded = self.model.decoder(
                    token_ids[:, chunk_start : chunk_start + chunk_size],
                    jnp.arange(chunk_start, chunk_start + chunk_size, dtype=jnp.int32)[None],
                    state=state,
                    return_updated_state=True,
                    lengths_without_padding=jnp.asarray([min(chunk_size, length - chunk_start)], dtype=jnp.int32),
                    forward_pass_config=config,
                    keychain=keychain,
                )
                assert decoded.updated_state is not None
                state = decoded.updated_state
                # Only continuation rows leave the device; copying full-vocab logits of prompt rows dominated runtime.
                chunk_output_start = max(0, output_start - chunk_start)
                chunk_output_end = min(chunk_size, length - chunk_start)
                if chunk_output_start < chunk_output_end:
                    chunks.append(np.asarray(decoded.logits[0, chunk_output_start:chunk_output_end], dtype=np.float16))
            result.append(np.concatenate(chunks))
        return result
