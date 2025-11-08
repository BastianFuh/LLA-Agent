from frontend.tabs.messages.en import ENMessages
from frontend.tabs.messages.messages import Messages


class MessageManager:
    """Manager for a messanges class to handle localization in the application."""

    _instance = None  # class-level variable

    def __init__(self):
        self._messages = ENMessages()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def setMessages(self, messages: Messages):
        self._messages = messages

    def getMessages(self) -> Messages:
        return self._messages
