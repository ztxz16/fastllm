import os
import sys

from PySide6.QtCore import QUrl, QLocale
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from i18n.lang_manager import LangManager


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_qml_dir(style: str = "default") -> str:
    return os.path.join(BASE_DIR, "qml", style)


def create_app(argv=None, style: str = "default"):
    if argv is None:
        argv = sys.argv
    app = QGuiApplication(argv)
    app.setOrganizationName("Fastllm")
    app.setApplicationName("Fastllm Studio")

    lang_mgr = LangManager(app)
    system_locale = QLocale.system().name()
    lang_mgr.load_language(system_locale)

    engine = QQmlApplicationEngine()

    from app.viewmodels.model_repo_vm import ModelRepoViewModel
    from app.viewmodels.model_market_vm import ModelMarketViewModel
    from app.viewmodels.chat_vm import ChatViewModel

    repo_vm = ModelRepoViewModel()
    market_vm = ModelMarketViewModel(repo_vm.store)
    chat_vm = ChatViewModel(repo_vm.server_manager)

    ctx = engine.rootContext()
    ctx.setContextProperty("repoViewModel", repo_vm)
    ctx.setContextProperty("marketViewModel", market_vm)
    ctx.setContextProperty("chatViewModel", chat_vm)
    ctx.setContextProperty("langManager", lang_mgr)

    qml_dir = get_qml_dir(style)
    engine.addImportPath(qml_dir)

    main_qml = os.path.join(qml_dir, "Main.qml")
    engine.load(QUrl.fromLocalFile(main_qml))

    if not engine.rootObjects():
        print("Failed to load QML")
        sys.exit(-1)

    window = engine.rootObjects()[0]
    screen = window.screen()
    geo = screen.availableGeometry()
    x = (geo.width() - window.width()) // 2 + geo.x()
    y = (geo.height() - window.height()) // 2 + geo.y()
    window.setX(x)
    window.setY(y)

    return app, engine, repo_vm, market_vm, chat_vm
