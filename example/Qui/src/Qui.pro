QT          += core gui svg widgets network
CONFIG += c++17

APP_NAME = Qui
DESTDIR     = ../bin


SOURCES += \
    Qui.cpp \
    main.cpp

HEADERS += \
    Qui.h \
    resource.h

FORMS += \
    Qui.ui

TRANSLATIONS = qui_cn.ts
