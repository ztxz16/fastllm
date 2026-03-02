import os
from PySide6.QtCore import QObject, QTranslator, QLocale, Slot, Signal, Property


I18N_DIR = os.path.dirname(os.path.abspath(__file__))

SUPPORTED_LANGUAGES = {
    "zh_CN": "简体中文",
    "en_US": "English",
}


class LangManager(QObject):
    languageChanged = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self._app = app
        self._translator = QTranslator(self)
        self._current_lang = ""

    @Slot(str)
    def load_language(self, locale_name: str):
        lang = self._resolve_locale(locale_name)
        if lang == self._current_lang:
            return

        self._app.removeTranslator(self._translator)
        self._translator = QTranslator(self)

        qm_file = os.path.join(I18N_DIR, f"{lang}.qm")
        if os.path.exists(qm_file) and self._translator.load(qm_file):
            self._app.installTranslator(self._translator)

        self._current_lang = lang
        self.languageChanged.emit()

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
