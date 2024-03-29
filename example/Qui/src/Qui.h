#pragma once

#include <QtWidgets/QWidget>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QMessageBox>
#include "ui_Qui.h"
#include <QtNetwork>
#include <QKeyEvent>

class Qui : public QWidget
{
    Q_OBJECT

public:
    Qui(QWidget *parent = nullptr);
    ~Qui() {};

private slots:
    void clearChat();
    void wirteToFile();
    void sendAI();
    void onReset();

    void onPathSelectModel();
    void onPathSelectFlm();

    void onDeviceCheck();

    void runModelWithSetting();
    void stopModelRuning();

    bool eventFilter(QObject *obj, QEvent *event) override;

    void readData();
    void finishedProcess();
    void errorProcess();

private:
    QProcess *process;
    void closeEvent(QCloseEvent *event) override;

private:
    Ui::QuiClass ui;

    QString modelPath = "";
    QString flmPath = "";
    void updateModelList();
    bool inited = false;
};
