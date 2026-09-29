import base64
import hashlib
import json
from dataclasses import dataclass

from cryptography.fernet import Fernet
from datasets import load_dataset
from lalamo.models.chat_codec import AssistantMessage, Message, ReasoningEffort, ToolSchema, parse_hf_message


@dataclass(frozen=True)
class Conversation:
    messages: list[Message]
    tools: list[ToolSchema] | None
    reasoning_effort: ReasoningEffort


def load_conversations(dataset_url: str) -> list[Conversation]:
    dataset = load_dataset(dataset_url, split="train")
    fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(b"mirai labs").digest()))
    conversations = [
        Conversation(
            [parse_hf_message(message) for message in json.loads(fernet.decrypt(messages))],
            json.loads(fernet.decrypt(tools)),
            ReasoningEffort(reasoning_effort),
        )
        for messages, tools, reasoning_effort in zip(
            dataset["messages"], dataset["tools"], dataset["reasoning_effort"], strict=True
        )
    ]
    if not conversations or any(
        not conversation.messages or isinstance(conversation.messages[-1], AssistantMessage)
        for conversation in conversations
    ):
        raise ValueError("every conversation must end in a non-assistant message")
    return conversations
