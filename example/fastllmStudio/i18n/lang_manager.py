import os
import sys
import subprocess
from PySide6.QtCore import QObject, QTranslator, QLocale, Slot, Signal, Property


I18N_DIR = os.path.dirname(os.path.abspath(__file__))

SUPPORTED_LANGUAGES = {
    "zh_CN": "简体中文",
    "en_US": "English",
}


def _compile_ts_if_needed(lang: str):
    """Compile .ts to .qm if the .qm is missing or older than the .ts."""
    ts_file = os.path.join(I18N_DIR, f"{lang}.ts")
    qm_file = os.path.join(I18N_DIR, f"{lang}.qm")
    if not os.path.exists(ts_file):
        return
    if os.path.exists(qm_file) and os.path.getmtime(qm_file) >= os.path.getmtime(ts_file):
        return
    commands = [
        ["lrelease", ts_file, "-qm", qm_file],
        ["pyside6-lrelease", ts_file, "-qm", qm_file],
    ]

    # Some PySide6 environments don't expose `pyside6-lrelease` in PATH,
    # but still ship an internal `lrelease` binary under site-packages/PySide6.
    try:
        import PySide6
        pyside_lrelease = os.path.join(os.path.dirname(PySide6.__file__), "lrelease")
        commands.append([pyside_lrelease, ts_file, "-qm", qm_file])
    except Exception:
        pass

    # Last fallback for uncommon setups.
    commands.append([sys.executable, "-m", "PySide6.scripts.pyside_tool", "lrelease", ts_file, "-qm", qm_file])

    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            continue


class LangManager(QObject):
    languageChanged = Signal()

    def __init__(self, app, engine=None, parent=None):
        super().__init__(parent)
        self._app = app
        self._engine = engine
        self._translator = QTranslator(self)
        self._current_lang = ""

    def set_engine(self, engine):
        self._engine = engine

    @Slot(str)
    def load_language(self, locale_name: str):
        lang = self._resolve_locale(locale_name)
        if lang == self._current_lang:
            return

        self._app.removeTranslator(self._translator)
        self._translator = QTranslator(self)

        _compile_ts_if_needed(lang)

        qm_file = os.path.join(I18N_DIR, f"{lang}.qm")
        if os.path.exists(qm_file) and self._translator.load(qm_file):
            self._app.installTranslator(self._translator)

        self._current_lang = lang
        self.languageChanged.emit()

        if self._engine is not None:
            self._engine.setUiLanguage(lang)
            self._engine.retranslate()

    @Slot(result=list)
    def availableLanguages(self):
        return list(SUPPORTED_LANGUAGES.keys())

    @Slot(str, result=str)
    def languageDisplayName(self, lang_code):
        return SUPPORTED_LANGUAGES.get(lang_code, lang_code)

    def _get_current_language(self):
        return self._current_lang

    currentLanguage = Property(str, _get_current_language, notify=languageChanged)

    @staticmethod
    def _resolve_locale(locale_name: str) -> str:
        if locale_name in SUPPORTED_LANGUAGES:
            return locale_name
        prefix = locale_name.split("_")[0] if "_" in locale_name else locale_name
        for key in SUPPORTED_LANGUAGES:
            if key.startswith(prefix):
                return key
        return "en_US"
