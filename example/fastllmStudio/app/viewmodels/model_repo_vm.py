import os
from PySide6.QtCore import QObject, Signal, Slot, Property, QAbstractListModel, Qt, QModelIndex

from app.models.model_store import ModelStore
from app.models.model_item import ModelItem, ModelLaunchConfig
from app.models.server_manager import ServerManager


class ModelListModel(QAbstractListModel):
    NameRole = Qt.ItemDataRole.UserRole + 1
    PathRole = Qt.ItemDataRole.UserRole + 2
    ModelIdRole = Qt.ItemDataRole.UserRole + 3
    SourceRole = Qt.ItemDataRole.UserRole + 4
    RunningRole = Qt.ItemDataRole.UserRole + 5
    PortRole = Qt.ItemDataRole.UserRole + 6
    DeviceRole = Qt.ItemDataRole.UserRole + 7
    ThreadsRole = Qt.ItemDataRole.UserRole + 8
    DtypeRole = Qt.ItemDataRole.UserRole + 9
    ConfigPortRole = Qt.ItemDataRole.UserRole + 10

    def __init__(self, store: ModelStore, server_mgr: ServerManager, parent=None):
        super().__init__(parent)
        self._store = store
        self._server_mgr = server_mgr
        self._items = store.get_all()

    def reload(self):
        self.beginResetModel()
        self._items = self._store.get_all()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._items)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._items):
            return None
        item = self._items[index.row()]
        if role == self.NameRole:
            return item.name
        if role == self.PathRole:
            return item.path
        if role == self.ModelIdRole:
            return item.model_id
        if role == self.SourceRole:
            return item.source
        if role == self.RunningRole:
            return self._server_mgr.is_running(item.model_id)
        if role == self.PortRole:
            p = self._server_mgr.get_port(item.model_id)
            return p if p else 0
        if role == self.DeviceRole:
            return item.launch_config.device
        if role == self.ThreadsRole:
            return item.launch_config.threads
        if role == self.DtypeRole:
            return item.launch_config.dtype
        if role == self.ConfigPortRole:
            return item.launch_config.port
        return None

    def roleNames(self):
        return {
            self.NameRole: b"name",
            self.PathRole: b"path",
            self.ModelIdRole: b"modelId",
            self.SourceRole: b"source",
            self.RunningRole: b"running",
            self.PortRole: b"port",
            self.DeviceRole: b"device",
            self.ThreadsRole: b"threads",
            self.DtypeRole: b"dtype",
            self.ConfigPortRole: b"configPort",
        }


class ModelRepoViewModel(QObject):
    modelsChanged = Signal()
    modelStarted = Signal(str, int, arguments=["modelId", "port"])
    modelStopped = Signal(str, arguments=["modelId"])

    def __init__(self, parent=None):
        super().__init__(parent)
        self._store = ModelStore()
        self._server_manager = ServerManager()
        self._list_model = ModelListModel(self._store, self._server_manager)

    @Property(QObject, constant=True)
    def modelList(self):
        return self._list_model

    @property
    def store(self):
        return self._store

    @property
    def server_manager(self):
        return self._server_manager

    @Slot(str, str)
    def addLocalModel(self, name, path):
        if not name:
            name = os.path.basename(path)
        self._store.add_model(name=name, path=path, source="local")
        self._list_model.reload()
        self.modelsChanged.emit()

    @Slot(str, str, str)
    def addDownloadedModel(self, name, path, source):
        self._store.add_model(name=name, path=path, source=source)
        self._list_model.reload()
        self.modelsChanged.emit()

    @Slot(str)
    def removeModel(self, model_id):
        self._server_manager.stop_server(model_id)
        self._store.remove_model(model_id)
        self._list_model.reload()
        self.modelsChanged.emit()

    @Slot(str)
    def startModel(self, model_id):
        item = self._store.get_by_id(model_id)
        if item is None:
            return
        port = self._server_manager.start_server(item)
        self._list_model.reload()
        self.modelsChanged.emit()
        self.modelStarted.emit(model_id, port)

    @Slot(str)
    def stopModel(self, model_id):
        self._server_manager.stop_server(model_id)
        self._list_model.reload()
        self.modelsChanged.emit()
        self.modelStopped.emit(model_id)

    @Slot(str, str, int, str, int)
    def updateLaunchConfig(self, model_id, device, threads, dtype, port):
        config = ModelLaunchConfig(
            device=device,
            threads=threads,
            dtype=dtype,
            port=port,
        )
        self._store.update_launch_config(model_id, config)
        self._list_model.reload()

    def shutdown(self):
        self._server_manager.shutdown_all()
