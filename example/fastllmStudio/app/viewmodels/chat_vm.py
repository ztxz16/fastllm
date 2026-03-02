from PySide6.QtCore import (
    QObject, Signal, Slot, Property,
    QAbstractListModel, Qt, QModelIndex, QThread,
)

from app.models.chat_client import ChatClient
from app.models.server_manager import ServerManager


class StreamWorker(QThread):
    tokenReceived = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, client: ChatClient, messages: list,
                 temperature: float, top_p: float, top_k: int,
                 repeat_penalty: float, parent=None):
        super().__init__(parent)
        self._client = client
        self._messages = messages
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._repeat_penalty = repeat_penalty

    def run(self):
        try:
            for token in self._client.stream_chat(
                messages=self._messages,
                temperature=self._temperature,
                top_p=self._top_p,
                top_k=self._top_k,
                repeat_penalty=self._repeat_penalty,
            ):
                self.tokenReceived.emit(token)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class MessageListModel(QAbstractListModel):
    RoleRole = Qt.ItemDataRole.UserRole + 1
    ContentRole = Qt.ItemDataRole.UserRole + 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._messages = []

    def rowCount(self, parent=QModelIndex()):
        return len(self._messages)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._messages):
            return None
        msg = self._messages[index.row()]
        if role == self.RoleRole:
            return msg["role"]
        if role == self.ContentRole:
            return msg["content"]
        return None

    def roleNames(self):
        return {
            self.RoleRole: b"role",
            self.ContentRole: b"content",
        }

    def append_message(self, role: str, content: str):
        self.beginInsertRows(QModelIndex(), len(self._messages), len(self._messages))
        self._messages.append({"role": role, "content": content})
        self.endInsertRows()

    def update_last(self, content: str):
        if not self._messages:
            return
        idx = len(self._messages) - 1
        self._messages[idx]["content"] = content
        model_idx = self.index(idx, 0)
        self.dataChanged.emit(model_idx, model_idx, [self.ContentRole])

    def clear_all(self):
        self.beginResetModel()
        self._messages = []
        self.endResetModel()

    def to_api_messages(self):
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]


class ChatViewModel(QObject):
    messagesChanged = Signal()
    streamingChanged = Signal()
    runningModelsChanged = Signal()
    streamError = Signal(str, arguments=["error"])

    def __init__(self, server_manager: ServerManager, parent=None):
        super().__init__(parent)
        self._server_manager = server_manager
        self._message_model = MessageListModel()
        self._streaming = False
        self._worker = None
        self._current_assistant_text = ""

        self._temperature = 1.0
        self._top_p = 1.0
        self._top_k = 0
        self._repeat_penalty = 1.0

        self._selected_model_id = ""

    @Property(QObject, constant=True)
    def messageList(self):
        return self._message_model

    def _get_streaming(self):
        return self._streaming

    streaming = Property(bool, _get_streaming, notify=streamingChanged)

    @Slot(result="QVariantList")
    def getRunningModels(self):
        return self._server_manager.get_running_models()

    @Slot(str)
    def selectModel(self, model_id):
        self._selected_model_id = model_id

    @Slot(float)
    def setTemperature(self, val):
        self._temperature = val

    @Slot(float)
    def setTopP(self, val):
        self._top_p = val

    @Slot(int)
    def setTopK(self, val):
        self._top_k = val

    @Slot(float)
    def setRepeatPenalty(self, val):
        self._repeat_penalty = val

    @Slot(str)
    def sendMessage(self, text):
        if not text.strip() or self._streaming:
            return
        if not self._selected_model_id:
            return

        port = self._server_manager.get_port(self._selected_model_id)
        model_name = self._server_manager.get_model_name(self._selected_model_id)
        if port is None or model_name is None:
            return

        self._message_model.append_message("user", text.strip())
        self._message_model.append_message("assistant", "")
        self._current_assistant_text = ""
        self.messagesChanged.emit()

        client = ChatClient(port, model_name)
        api_messages = self._message_model.to_api_messages()[:-1]

        self._streaming = True
        self.streamingChanged.emit()

        self._worker = StreamWorker(
            client, api_messages,
            self._temperature, self._top_p, self._top_k, self._repeat_penalty,
        )
        self._worker.tokenReceived.connect(self._on_token)
        self._worker.finished.connect(self._on_stream_done)
        self._worker.error.connect(self._on_stream_error)
        self._worker.start()

    def _on_token(self, token: str):
        self._current_assistant_text += token
        self._message_model.update_last(self._current_assistant_text)

    def _on_stream_done(self):
        self._streaming = False
        self.streamingChanged.emit()
        self.messagesChanged.emit()

    def _on_stream_error(self, err: str):
        self._current_assistant_text += f"\n[Error: {err}]"
        self._message_model.update_last(self._current_assistant_text)
        self._streaming = False
        self.streamingChanged.emit()
        self.streamError.emit(err)

    @Slot()
    def clearChat(self):
        self._message_model.clear_all()
        self.messagesChanged.emit()

    @Slot()
    def refreshRunningModels(self):
        self.runningModelsChanged.emit()
