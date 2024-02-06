from abc import ABC

from mocogpt.core.chat_completions import Chat
from mocogpt.core.embeddings import Embeddings


class GptServer(ABC):
    def __init__(self, chat: Chat, embeddings: Embeddings):
        self._chat = chat
        self._embeddings = embeddings

    @property
    def chat(self) -> Chat:
        return self._chat

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings
