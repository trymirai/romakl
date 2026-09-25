from typing import cast

from datasets import load_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.models.chat_codec import AssistantMessage, Message


def load_conversations(dataset_url: str) -> list[list[Message]]:
    dataset = load_dataset(dataset_url)
    conversations = [
        [HFMessage.from_dict(message).as_message() for message in row["messages"]]
        for index in range(len(dataset))
        for row in (dataset[index],)
    ]
    if not conversations or any(
        not conversation or isinstance(conversation[-1], AssistantMessage) for conversation in conversations
    ):
        raise ValueError("every conversation must end in a non-assistant message")
    return conversations
