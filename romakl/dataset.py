import base64
import hashlib
import json
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from lalamo.models.chat_codec import AssistantMessage, Message, ReasoningEffort, ToolSchema, parse_hf_message


@dataclass(frozen=True)
class Conversation:
    messages: list[Message]
    tools: list[ToolSchema] | None
    reasoning_effort: ReasoningEffort
    source: str
    domain: str


def try_decrypting(value: str | list[dict] | None, fernet: Fernet) -> list[dict] | None:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(fernet.decrypt(value))
    except InvalidToken as error:
        error.add_note("Decryption with the default key failed; please decrypt the dataset before passing it in.")
        raise


def load_conversations(dataset_url: str) -> list[Conversation]:
    try:
        dataset = load_dataset(dataset_url, split="train")
    except DatasetNotFoundError as error:
        error.add_note("Maybe you are not logged into Hugging Face? Run `hf auth login`.")
        raise
    fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(b"mirai labs").digest()))
    conversations = [
        Conversation(
            [parse_hf_message(message) for message in try_decrypting(messages, fernet)],
            try_decrypting(tools, fernet),
            ReasoningEffort(reasoning_effort),
            source,
            domain,
        )
        for messages, tools, reasoning_effort, source, domain in zip(
            dataset["messages"],
            dataset["tools"],
            dataset["reasoning_effort"],
            dataset["source"],
            dataset["domain"],
            strict=True,
        )
    ]
    if not conversations or any(
        not conversation.messages or isinstance(conversation.messages[-1], AssistantMessage)
        for conversation in conversations
    ):
        raise ValueError("every conversation must end in a non-assistant message")
    return conversations
