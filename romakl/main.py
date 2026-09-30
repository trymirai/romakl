# ruff: noqa: I001
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Annotated

import cattrs
from loguru import logger
from typer import Option, Typer

from .model import Model
from .vendors.lalamo import LalamoModel
from .dataset import load_conversations
from .scoring import mean_score, score
from .rollouts import Rollout, hash_folder_with_edittimes, load_rollouts, save_rollouts

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def run(
    reference: Annotated[Path, Option(help="Lalamo reference model path")],
    students: Annotated[list[str], Option("--student", help="Student as engine:path; repeat for several")],
    dataset: Annotated[str, Option(help="Hugging Face conversation dataset")] = "trymirai/kl-mixture",
    cache_path: Annotated[Path, Option(help="Reference rollout cache")] = Path("teacher-cache"),
) -> None:
    conversations = load_conversations(dataset)
    logger.info("Loaded {} conversations", len(conversations))

    reference_model = LalamoModel.load(reference.expanduser())
    generation_config = reference_model.model.config.generation_config
    logger.info("Loaded the reference")

    reasoning_config = reference_model.codec.config.reasoning_config
    supported_efforts = reasoning_config.reasoning_effort_to_field_value if reasoning_config else {}
    for index, conversation in enumerate(conversations):
        if conversation.reasoning_effort not in supported_efforts:
            effort = conversation.reasoning_effort
            logger.warning("{} asks for unsupported {} reasoning; using the default", conversation.source, effort)
            conversations[index] = replace(conversation, reasoning_effort=None)

    prompts = reference_model.tokenize(conversations)
    kept = [index for index, prompt in enumerate(prompts) if len(prompt) <= 32_768]
    conversations = [conversations[index] for index in kept]
    prompts = [prompts[index] for index in kept]
    logger.info("Kept {} prompts of at most 32k tokens", len(prompts))

    token_limit = 1024
    cache_key = (
        hash_folder_with_edittimes(reference_model.path),
        prompts,
        cattrs.unstructure(generation_config),
        token_limit,
    )
    rollout_path = cache_path / f"{sha256(json.dumps(cache_key).encode()).hexdigest()}.json"
    reference_rollouts: list[Rollout] | None = load_rollouts(rollout_path)
    if reference_rollouts is None:
        logger.info("Generating {} rollouts", len(prompts))
        reference_rollouts = reference_model.generate(prompts, generation_config, seed=0, token_limit=token_limit)
        save_rollouts(rollout_path, reference_rollouts)

    logger.info("Computing teacher logits")
    teacher_logits = reference_model.logits(reference_rollouts)
    scores, sizes = {}, {}
    for student_name in students:
        logger.info("Scoring {}", student_name)
        student = Model.init(student_name, reference_model.codec.config)
        if student.tokenize(conversations) != prompts:
            raise ValueError(f"{student_name} tokenization differs from the reference")
        scores[student_name] = score(student, teacher_logits, reference_rollouts, generation_config)
        sizes[student_name] = student.generator_bytes() / 1e9
        del student  # Release the weights before the next student loads.

    width = max(len(Path(name).name) for name in scores)
    for student_name, student_scores in scores.items():
        print(f"{Path(student_name).name:<{width}}  {mean_score(student_scores)}, {sizes[student_name]:.2f} GB")
    for index, (conversation, rollout) in enumerate(zip(conversations, reference_rollouts, strict=True)):
        generated = sum(rollout.mask)
        prompt_tokens = len(rollout.tokens) - generated
        print(f"{index},{conversation.domain}/{conversation.source}/prompt {prompt_tokens}, generated {generated}:")
        for student_name, student_scores in scores.items():
            print(f"\t{Path(student_name).name:<{width}}  {student_scores[index]}")


if __name__ == "__main__":
    app()
