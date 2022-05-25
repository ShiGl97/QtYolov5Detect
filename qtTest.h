#pragma once
#include <gdal_priv.h>
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp> 
#include <qlabel.h> 
#include <iostream>
#include "ogrsf_frmts.h"
#include <QtWidgets/QMainWindow>
#include "ui_qtTest.h"
#include "yolo.h"


class Output;
class Yolo;
class qtTest : public QMainWindow
{
	
	Q_OBJECT

public:
	//friend class Yolo;
	//friend class Output;
	qtTest(QWidget *parent = Q_NULLPTR);

private:
	QString imageInfo1;
	QString imageInfo2;
	QString imageInfo3;
	QString imageInfo4, imageInfo5;
	QLabel* label;
	QLabel* label_2;
	QLabel* label_3;
	Ui::qtTestClass ui;
public:
	QLabel* imgLabel;
	//Mat image;
private slots:
	void on_Progress_clicked();
	//void getImagineInfo();
	void on_OpenFig_clicked();
	//����ʽon_�ؼ���_clicked����������QT��Ĭ����ɺ����Ͱ�ť���������ӣ���������������Ļ���ȥ�����źŲۺ���
	
	void on_CnnFig_clicked();
	//void on_CnnFigSave_clicked();
	void on_CnnResult_clicked();
};
