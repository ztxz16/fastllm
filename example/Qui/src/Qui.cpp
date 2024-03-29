#include "Qui.h"

Qui::Qui(QWidget *parent)
    : QWidget(parent)
{
    ui.setupUi(this);
    ui.textEditUser->installEventFilter(this);
    ui.textEditAI->setReadOnly(true);
    ui.textEditUser->setFocus();

    modelPath = QCoreApplication::applicationDirPath() + "/model";
    flmPath = QCoreApplication::applicationDirPath() + "/fastllm_cpu.exe";

    QFile file(QCoreApplication::applicationDirPath() + "/path.txt");

    if (file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&file);

        if (!in.atEnd())
        {
            QString line = in.readLine();

            if (line != "")
            {
                modelPath = line;
            }
        }

        if (!in.atEnd())
        {
            QString line = in.readLine();

            if (line != "")
            {
                flmPath = line;
            }
        }

        file.close();
    }


    ui.edPathModel->setText(modelPath);
    ui.edPathFlm->setText(flmPath);

    updateModelList();

    ui.gbDeviceMap->setChecked(false);
    ui.btnStopModel->setEnabled(false);
}

void
Qui::clearChat()
{
    ui.textEditAI->clear();
}

void
Qui::wirteToFile()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save File", "Qui", "Markdown (*.md)");

    if (!fileName.isEmpty())
    {
        QFile file(fileName);

        if (file.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            QTextStream out(&file);
            QString textData = ui.textEditAI->toPlainText();
            out << textData;
            file.close();
        }
        else
        {
            QMessageBox::about(this, tr("Write file error"), tr("Unable to write to specified location!"));
        }
    }
}

void
Qui::sendAI()
{
    if (ui.textEditUser->toPlainText().isEmpty())
    {
        ui.textEditUser->setPlaceholderText(tr("The input message cannot be empty!"));
        return;
    }

    ui.textEditUser->setPlaceholderText(tr("The input message cannot be empty!"));

    double topP = ui.sbTopP->value();                   //采样参数top_p
    int topK = ui.sbTopK->value();                      //采样参数top_k
    double temperature = ui.sbTemperature->value();     //采样参数温度，越高结果越不固定
    double repeatPenalty = ui.sbRepeatPenalty->value(); //采样参数重复惩罚

    QString cmd = "generationConfig\r\n";
    process->write(cmd.toStdString().data());

    cmd = "--top_p " + QString::number(topP) +
          " --top_k " + QString::number(topK) +
          " --temperature " + QString::number(temperature) +
          " --repeat_penalty " + QString::number(repeatPenalty) +
          "\r\n";

    process->write(cmd.toStdString().data());

    cmd = ui.textEditUser->toPlainText();
    ui.textEditAI->append(QString("<font color=blue>用户:%1</font>").arg(cmd));
    ui.textEditAI->append("\n");
    ui.textEditUser->clear();
    cmd += "\r\n";
    process->write(cmd.toStdString().data());
    ui.textEditAI->moveCursor(QTextCursor::End);
}

void
Qui::onReset()
{
    QString cmd = "reset";
    ui.textEditAI->append(QString("<font color=blue>用户:%1</font>").arg(cmd));
    ui.textEditAI->append("\n");
    ui.textEditUser->clear();
    cmd += "\r\n";
    process->write(cmd.toStdString().data());
    ui.textEditAI->moveCursor(QTextCursor::End);
}

void
Qui::onPathSelectModel()
{
    QString dir = QFileDialog::getExistingDirectory(nullptr, QObject::tr("Open Directory"), modelPath);

    if (dir.isEmpty())
    {
        dir = QCoreApplication::applicationDirPath() + "/model";
    }

    ui.edPathModel->setText(dir);
    modelPath = dir;
    updateModelList();
}

void
Qui::onPathSelectFlm()
{
    QString filePath = QFileDialog::getOpenFileName(nullptr, QObject::tr("Open File"),
                       flmPath, QObject::tr("All Files (*)"));

    if (filePath.isEmpty())
    {
        filePath = QCoreApplication::applicationDirPath() + "/Qui.exe";
    }

    ui.edPathFlm->setText(filePath);
    flmPath = filePath;
}

void
Qui::onDeviceCheck()
{
    if (ui.ckDev2->isChecked())
    {
        ui.cbDev2->setEnabled(true);
        ui.sbDev2->setEnabled(true);
    }
    else
    {
        ui.cbDev2->setEnabled(false);
        ui.sbDev2->setEnabled(false);
    }

    if (ui.ckDev3->isChecked())
    {
        ui.cbDev3->setEnabled(true);
        ui.sbDev3->setEnabled(true);
    }
    else
    {
        ui.cbDev3->setEnabled(false);
        ui.sbDev3->setEnabled(false);
    }

    if (ui.ckDev4->isChecked())
    {
        ui.cbDev4->setEnabled(true);
        ui.sbDev4->setEnabled(true);
    }
    else
    {
        ui.cbDev4->setEnabled(false);
        ui.sbDev4->setEnabled(false);
    }
}

void
Qui::runModelWithSetting()
{
    ui.btnApplyModel->setEnabled(false);
    ui.btnStopModel->setEnabled(true);
    ui.gbSetting->setEnabled(false);

    QStringList arguments;
    arguments << "-p" << modelPath + "/" + ui.cbModelList->currentText();//模型文件的路径
    arguments << "-t" << QString::number(ui.sbThreads->value());//使用的线程数量

    if (ui.gbDeviceMap->isChecked())
    {
        int numDev = 2;
        QStringList devStr;

        devStr << ui.cbDev0->currentText() << QString::number(ui.sbDev0->value());
        devStr << ui.cbDev1->currentText() << QString::number(ui.sbDev1->value());

        if (ui.ckDev2->isChecked())
        {
            ++numDev;
            devStr << ui.cbDev2->currentText() << QString::number(ui.sbDev2->value());
        }

        if (ui.ckDev3->isChecked())
        {
            ++numDev;
            devStr << ui.cbDev3->currentText() << QString::number(ui.sbDev3->value());
        }

        if (ui.ckDev4->isChecked())
        {
            ++numDev;
            devStr << ui.cbDev4->currentText() << QString::number(ui.sbDev4->value());
        }

        int numCuda = 0;

        for (int i = 0; i < devStr.size(); ++i)
        {
            if (devStr[i].contains("cuda"))
            {
                ++numCuda;
            }
        }

        if (numCuda <= 1)
        {
            for (int i = 0; i < devStr.size(); ++i)
            {
                if (devStr[i].contains("cuda"))
                {
                    devStr[i] = "cuda";
                }
            }
        }

        arguments << "--device_map" << QString::number(numDev);

        for (int i = 0; i < devStr.size(); ++i)
        {
            arguments << devStr[i];
        }
    }

    if (ui.cbLow->checkState())
    {
        arguments << "-l";
    }

    process = new QProcess(this);
    process->setProcessChannelMode(QProcess::MergedChannels);

    connect(process, SIGNAL(readyRead()), this, SLOT(readData()));
    connect(process, SIGNAL(readyReadStandardOutput()), this, SLOT(readData()));
    connect(process, SIGNAL(errorOccurred(QProcess::ProcessError)), this, SLOT(errorProcess()));
    connect(process, SIGNAL(finished(int)), this, SLOT(finishedProcess()));

    process->start(flmPath, arguments);
    process->waitForStarted();
    process->waitForFinished();
}

void
Qui::stopModelRuning()
{
    process->kill();
    ui.btnApplyModel->setEnabled(true);
    ui.btnStopModel->setEnabled(false);
    ui.gbSetting->setEnabled(true);

    inited = false;
}

bool
Qui::eventFilter(QObject *obj, QEvent *event)
{
    if (obj == ui.textEditUser && event->type() == QEvent::KeyPress)
    {
        QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);

        if (keyEvent->key() == Qt::Key_Return || keyEvent->key() == Qt::Key_Enter)
        {
            if (ui.btnSend->isEnabled() == true)
            {
                ui.btnSend->clicked();
                return true; // Return true to indicate that the event has been handled
            }
        }
    }

    return QObject::eventFilter(obj, event);
}


QString
convToUtf8(const QByteArray &ba)
{
    QTextCodec::ConverterState state;
    QTextCodec *codec = QTextCodec::codecForName("UTF-8");
    QString text = codec->toUnicode(ba.constData(), ba.size(), &state);

    if (state.invalidChars > 0)
    {
        text = QTextCodec::codecForName("GBK")->toUnicode(ba);
    }
    else
    {
        text = ba;
    }

    return text;
}


void
Qui::readData()
{
    QByteArray text = process->readAll();

    if (inited == false)
    {
        QString strs = convToUtf8(text);

        if (strs.contains("finish"))
        {
            QStringList sList = strs.split("\n");
            ui.textEditAI->insertPlainText(sList[2]);
            ui.textEditAI->moveCursor(QTextCursor::End);
            inited = true;
        }
    }
    else
    {
        ui.textEditAI->insertPlainText(convToUtf8(text));
        ui.textEditAI->moveCursor(QTextCursor::End);
    }
}

void
Qui::finishedProcess()
{
    int flag = process->exitCode();
    stopModelRuning();
}

void
Qui::errorProcess()
{
    int err_code  = process->exitCode();
    QString err = process->errorString();

    ui.textEditAI->append(QString("error coed:%1").arg(err_code));
    ui.textEditAI->append(err);
}

void
Qui::closeEvent(QCloseEvent *event)
{
    QFile file(QCoreApplication::applicationDirPath() + "/path.txt");

    if (file.open(QIODevice::WriteOnly))
    {
        QTextStream out(&file);
        out << ui.edPathModel->text() + "\n";
        out << ui.edPathFlm->text() + "\n";
        file.close();
    }
}

void
Qui::updateModelList()
{
    ui.cbModelList->clear();
    QDir dir(modelPath);

    QStringList filters;
    filters << QString("*.flm");
    dir.setFilter(QDir::Files | QDir::Hidden | QDir::NoSymLinks);
    dir.setNameFilters(filters);

    dir.setSorting(QDir::Size | QDir::Reversed);

    QFileInfoList list = dir.entryInfoList();

    for (int i = 0; i < list.size(); ++i)
    {
        QFileInfo fileInfo = list.at(i);
        ui.cbModelList->addItem(fileInfo.fileName());
    }
}
