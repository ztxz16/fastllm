#include "Qui.h"
#include <QtWidgets/QApplication>

int
main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QTranslator tran;

    if (tran.load(QString("qui_cn.qm")))
    {
        a.installTranslator(&tran);
    }

    Qui w;
    w.show();
    return a.exec();
}
