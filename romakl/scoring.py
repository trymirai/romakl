# ruff: noqa: I001
from .model import StudentModel
from .rollouts import Rollout
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from lalamo.models.language_model import GenerationConfig


def expected_sampling_disagreement(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    rollout: Rollout,
    generation_config: GenerationConfig,
) -> jax.Array:
    policy = generation_config.default_policy()
    if not policy.has_count_penalties:
        output_tokens = len(student_logits)
        padding = ((0, -output_tokens % 256), (0, 0))
        student_rows = jnp.pad(jnp.asarray(student_logits, dtype=jnp.float32), padding)
        teacher_rows = jnp.pad(jnp.asarray(teacher_logits, dtype=jnp.float32), padding)
        student_probs = jax.nn.softmax(jax.vmap(policy.process_logits)(student_rows))
        teacher_probs = jax.nn.softmax(jax.vmap(policy.process_logits)(teacher_rows))
        return jnp.abs(student_probs - teacher_probs).sum(axis=-1)[:output_tokens] / 2

    output_start = len(rollout.tokens) - sum(rollout.mask)
    prompt = jnp.asarray(rollout.tokens[:output_start], dtype=jnp.int32)
    policy = policy.with_prompt_token_counts(prompt, jnp.asarray(output_start), student_logits.shape[-1])

    disagreements = []
    for student_row, teacher_row, token in zip(
        student_logits, teacher_logits, rollout.tokens[output_start:], strict=True
    ):
        student_probs = jax.nn.softmax(policy.process_logits(jnp.asarray(student_row, dtype=jnp.float32)))
        teacher_probs = jax.nn.softmax(policy.process_logits(jnp.asarray(teacher_row, dtype=jnp.float32)))
        disagreements.append(jnp.abs(student_probs - teacher_probs).sum() / 2)
        policy = policy.with_next_token_count(jnp.asarray(token))
    return jnp.stack(disagreements)


def kl_divergence(student_logits: jax.Array, teacher_logits: jax.Array) -> jax.Array:
    student_logp = jax.nn.log_softmax(student_logits.astype(jnp.float32))
    teacher_logp = jax.nn.log_softmax(teacher_logits.astype(jnp.float32))
    return (jnp.exp(teacher_logp) * (teacher_logp - student_logp)).sum(axis=-1)


def tv_divergence(student_logits: jax.Array, teacher_logits: jax.Array) -> jax.Array:
    student_probs = jax.nn.softmax(student_logits.astype(jnp.float32))
    teacher_probs = jax.nn.softmax(teacher_logits.astype(jnp.float32))
    return jnp.abs(student_probs - teacher_probs).sum(axis=-1) / 2


def top1_agreement(student_logits: jax.Array, teacher_logits: jax.Array) -> jax.Array:
    return student_logits.argmax(axis=-1) == teacher_logits.argmax(axis=-1)


@dataclass(frozen=True)
class Score:
    kl: float
    tv: float
    esd: float
    top1: float

    def __str__(self) -> str:
        return f"kl {self.kl:.4f}, tv {self.tv:.4f}, esd {self.esd:.4f}, top-1 {self.top1:.3f}"


def mean_score(scores: list[Score]) -> Score:
    return Score(
        kl=sum(score.kl for score in scores) / len(scores),
        tv=sum(score.tv for score in scores) / len(scores),
        esd=sum(score.esd for score in scores) / len(scores),
        top1=sum(score.top1 for score in scores) / len(scores),
    )


def score(
    student: StudentModel,
    all_teacher_logits: list[np.ndarray],
    rollouts: list[Rollout],
    generation_config: GenerationConfig,
) -> list[Score]:
    all_student_logits = student.logits(rollouts)

    result = []
    for rollout, host_teacher_logits, host_student_logits in zip(
        rollouts, all_teacher_logits, all_student_logits, strict=True
    ):
        teacher_logits = jnp.asarray(host_teacher_logits)
        student_logits = jnp.asarray(host_student_logits)
        esd = expected_sampling_disagreement(student_logits, teacher_logits, rollout, generation_config)
        result.append(
            Score(
                kl=float(kl_divergence(student_logits, teacher_logits).mean()),
                tv=float(tv_divergence(student_logits, teacher_logits).mean()),
                esd=float(esd.mean()),
                top1=float(top1_agreement(student_logits, teacher_logits).mean()),
            )
        )
    return result
