
#include "qtTest.h"
#include "yolo.h"
#include <QtWidgets/QApplication>
#include <qdebug.h>

using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    qtTest w;
    w.show();
    return a.exec();
}



