
#include "qtTest.h"
#include "yolo.h"
#include "qfiledialog.h"
#include <string.h>



using namespace std;
using namespace cv;
using namespace dnn;



qtTest::qtTest(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
}


void qtTest::on_Progress_clicked() {
    //ע���ļ���ʽ
    GDALAllRegister();
    GDALDataset* poDataset;
    QString filename;
    filename = QFileDialog::getOpenFileName(this,
        tr("ѡ��ͼ��"),
        "",
        tr("Images (*.png *.bmp *.jpg *.tif *.GIF )"));

    //Qstring����ת��Ϊconst char*

    std::string str = filename.toStdString();
    const char* pszFileName = str.c_str();
    //const char* pszFileName = "E:\\OpenCV_image\\d_SRC_00_00.tif";
    //��ֻ����ʽ��ͼ��
    poDataset = (GDALDataset*)GDALOpen(pszFileName, GA_ReadOnly);
    if (poDataset == NULL) {
        printf("��ʧ��!");
    }
    else {
        printf("�򿪳ɹ�!\n");

    }
    //��ȡͼ���ʽ��Ϣ
    std::cout << poDataset->GetDriver()->GetDescription() << std::endl;
    imageInfo1 = poDataset->GetDriver()->GetDescription();
    //���ͼ��Ĵ�С�Ͳ��θ���
    printf("ͼ���С��%dx%dx\n",
        poDataset->GetRasterXSize(), poDataset->GetRasterYSize());
    imageInfo2 = QString::number(poDataset->GetRasterXSize(), 10);
    imageInfo2.append("*");
    imageInfo2.append(QString::number(poDataset->GetRasterYSize(), 10));
    printf("���θ�����%d\n", poDataset->GetRasterCount());
    imageInfo3 = QString::number(poDataset->GetRasterCount());
    //���ͼ��ͶӰ��Ϣ
    if (poDataset->GetProjectionRef() != NULL) {
        printf("ͶӰ��Ϣ:'%s'\n", poDataset->GetProjectionRef());
    }
    //���ͼ�ε�����ͷֱ�����Ϣ
    double adfGeoTransform[6];
    if (poDataset->GetGeoTransform(adfGeoTransform) == CE_None) {
        printf("������Ϣ = (%.6f,%.6f)\n", adfGeoTransform[0], adfGeoTransform[3]);
        printf("�ֱ�����Ϣ = (%.6f,%.6f)\n", adfGeoTransform[1], adfGeoTransform[5]);

    }
    imageInfo4 = QString("%1").arg(adfGeoTransform[0]);
    imageInfo4.append(",");
    imageInfo4.append(QString("%1").arg(adfGeoTransform[3]));
    imageInfo5 = QString("%1").arg(adfGeoTransform[1]);
    imageInfo5.append(",");
    imageInfo5.append(QString("%1").arg(adfGeoTransform[5]));
    //imageInfo4 = adfGeoTransform[0] + adfGeoTransform[3];
    //imageInfo5 = adfGeoTransform[1] + adfGeoTransform[5];
    //��ȡդ�񲨶�
    GDALRasterBand* poBand;
    int nBlockXSize, nBlockYSize;
    int nGotMin, nGotMax;
    double adfMinMax[2];
    //��ȡ��һ������
    poBand = poDataset->GetRasterBand(1);
    //��ȡ�ò��ε������Сֵ�������ȡʧ�ܣ������ͳ��
    adfMinMax[0] = poBand->GetMinimum(&nGotMin);
    adfMinMax[1] = poBand->GetMaximum(&nGotMax);
    if (!(nGotMin && nGotMax)) {
        GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE, adfMinMax);
    }
    printf("������Сֵ = %.3fd, �������ֵ = %.3f\n", adfMinMax[0], adfMinMax[1]);

    if (poBand->GetOverviewCount() > 0) {
        printf("Band has %d overviews.\n", poBand->GetOverviewCount());
    }
    if (poBand->GetColorTable() != NULL) {
        printf("Band has a color table with %d entries.\n",
            poBand->GetColorTable()->GetColorEntryCount());
    }
    int nXSize = poBand->GetXSize();
    float* pafScanline = new float[nXSize];
    ui.lineEdit->setText(imageInfo1);
    ui.lineEdit_2->setText(imageInfo2);
    ui.lineEdit_3->setText(imageInfo3);
    ui.lineEdit_4->setText(imageInfo4);
    ui.lineEdit_5->setText(imageInfo5);
}


void qtTest::on_CnnResult_clicked()
{

    QString filename;
    filename = QFileDialog::getOpenFileName(this,
        tr("ѡ��ͼ��"),
        "",
        tr("Images (*.png *.bmp *.jpg *.tif *.GIF )"));
    if (filename.isEmpty())
    {
        return;
    }
    else
    {
        //QImage* img = new QImage;

        //if (!(img->load(filename))) //����ͼ��
        //{
        //	QMessageBox::information(this,
        //		tr("��ͼ��ʧ��"),
        //		tr("��ͼ��ʧ��!"));
        //	delete img;
        //	return;
        //}

        string str = filename.toStdString();  // ��filenameת��Ϊstring���ͣ�
        Mat image = imread(str);
        //image=imread(fileName.toLatin1().data);
        cvtColor(image, image, COLOR_BGR2RGB);
        cv::resize(image, image, Size(380, 490));
        QImage img = QImage((const unsigned char*)(image.data), image.cols, image.rows, QImage::Format_RGB888);


        label_3 = new QLabel();
        label_3->setPixmap(QPixmap::fromImage(img));
        label_3->resize(QSize(img.width(), img.height()));
        ui.scrollArea_3->setWidget(label_3);
    }
}


void qtTest::on_OpenFig_clicked()
{

    QString filename;
    filename = QFileDialog::getOpenFileName(this,
        tr("ѡ��ͼ��"),
        "",
        tr("Images (*.png *.bmp *.jpg *.tif *.GIF )"));
    if (filename.isEmpty())
    {
        return;
    }
    else
    {
        //QImage* img = new QImage;

        //if (!(img->load(filename))) //����ͼ��
        //{
        //	QMessageBox::information(this,
        //		tr("��ͼ��ʧ��"),
        //		tr("��ͼ��ʧ��!"));
        //	delete img;
        //	return;
        //}

        string str = filename.toStdString();  // ��filenameת��Ϊstring���ͣ�
        Mat image = imread(str);
        //image=imread(fileName.toLatin1().data);
        cvtColor(image, image, COLOR_BGR2RGB);
        cv::resize(image, image, Size(380, 490));
        QImage img = QImage((const unsigned char*)(image.data), image.cols, image.rows, QImage::Format_RGB888);


        label = new QLabel();
        label->setPixmap(QPixmap::fromImage(img));
        label->resize(QSize(img.width(), img.height()));
        ui.scrollArea->setWidget(label);


        //flip(image, image, 4);//��ת���� 0 ���·�ת��������ˮƽ��ת��������ˮƽ��ֱ����ת
        GaussianBlur(image, image, Size(5, 5), 15);//��˹ģ��
        //bilateralFilter(image, image, 0, 100, 10);//˫��ģ��
        QImage img1 = QImage((const unsigned char*)(image.data), image.cols, image.rows, QImage::Format_RGB888);
        label_2 = new QLabel();
        label_2->setPixmap(QPixmap::fromImage(img1));
        label_2->resize(QSize(img1.width(), img1.height()));
        ui.scrollArea_2->setWidget(label_2);



    }
}

bool Yolo::readModel(Net& net, string& netPath, bool isCuda = false) {
    try {
        net = readNetFromONNX(netPath);
    }
    catch (const std::exception&) {
        return false;
    }
    //cuda
    if (isCuda) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    //cpu
    else {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    return true;
}


bool Yolo::Detect(Mat& SrcImg, Net& net, vector<Output>& output) {
    Mat blob;
    int col = SrcImg.cols;
    int row = SrcImg.rows;
    int maxLen = MAX(col, row);
    Mat netInputImg = SrcImg.clone();
    if (maxLen > 1.2 * col || maxLen > 1.2 * row) {
        Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
        SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
        netInputImg = resizeImg;
    }
    //blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(104, 117, 123), true, false);
    blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0, 0), true, false);//���ѵ����δ��ͼƬ���м�ȥ��ֵ����������Ҫ����Ϊ���
    //blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(114, 114,114), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> netOutputImg;
    //vector<string> outputLayerName{"345","403", "461","output" };
    //net.forward(netOutputImg, outputLayerName[3]); //��ȡoutput�����
    net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
    std::vector<int> classIds;//���id����
    std::vector<float> confidences;//���ÿ��id��Ӧ���Ŷ�����
    std::vector<cv::Rect> boxes;//ÿ��id���ο�
    float ratio_h = (float)netInputImg.rows / netHeight;
    float ratio_w = (float)netInputImg.cols / netWidth;
    int net_width = className.size() + 5;  //������������������+5
    float* pdata = (float*)netOutputImg[0].data;
    for (int stride = 0; stride < 3; stride++) {    //stride
        int grid_x = (int)(netWidth / netStride[stride]);
        int grid_y = (int)(netHeight / netStride[stride]);
        for (int anchor = 0; anchor < 3; anchor++) { //anchors
            const float anchor_w = netAnchors[stride][anchor * 2];
            const float anchor_h = netAnchors[stride][anchor * 2 + 1];
            for (int i = 0; i < grid_y; i++) {
                for (int j = 0; j < grid_y; j++) {
                    float box_score = Sigmoid(pdata[4]);//��ȡÿһ�е�box���к���ĳ������ĸ���
                    if (box_score > boxThreshold) {
                        //Ϊ��ʹ��minMaxLoc(),��85����������Mat����
                        cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
                        Point classIdPoint;
                        double max_class_socre;
                        minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                        max_class_socre = Sigmoid((float)max_class_socre);
                        if (max_class_socre > classThreshold) {
                            //rect [x,y,w,h]
                            float x = (Sigmoid(pdata[0]) * 2.f - 0.5f + j) * netStride[stride];  //x
                            float y = (Sigmoid(pdata[1]) * 2.f - 0.5f + i) * netStride[stride];   //y
                            float w = powf(Sigmoid(pdata[2]) * 2.f, 2.f) * anchor_w;   //w
                            float h = powf(Sigmoid(pdata[3]) * 2.f, 2.f) * anchor_h;  //h
                            int left = (x - 0.5 * w) * ratio_w;
                            int top = (y - 0.5 * h) * ratio_h;
                            classIds.push_back(classIdPoint.x);
                            confidences.push_back(max_class_socre * box_score);
                            boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
                        }
                    }
                    pdata += net_width;//ָ���Ƶ���һ��
                }
            }
        }
    }
    vector<int> nms_result;
    NMSBoxes(boxes, confidences, classThreshold, nmsThreshold, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Output result;
        result.id = classIds[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }

    if (output.size())
        return true;
    else
        return false;
}


void Yolo::drawPred(Mat& img, vector<Output> result, vector<Scalar> color) {
    for (int i = 0; i < result.size(); i++) {
        int left, top;
        left = result[i].box.x;
        top = result[i].box.y;
        int color_num = i;
        rectangle(img, result[i].box, color[result[i].id], 2, 8);
        //���Ŷȱ�ʾ
        string label = className[result[i].id] + ":" + to_string(result[i].confidence);

        int baseLine;
        //Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
        putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
    }
    imshow("res", img);
    //ʵ��ѡ��·���ı���
    cout << "�����뱣��·��:" << endl;
    string str;
    cin >> str;
    imwrite(str, img);
    waitKey();
    //destroyAllWindows();
}


void qtTest::on_CnnFig_clicked() {
    QString filename;
    filename = QFileDialog::getOpenFileName(this,
        tr("ѡ��ͼ��"),
        "",
        tr("Images (*.png *.bmp *.jpg *.tif *.GIF )"));

    //Qstring����ת��Ϊconst char*

    std::string str = filename.toStdString();
    const char* pszFileName = str.c_str();
    string img_path = pszFileName;
    string model_path = "D:/YOLOv5/yolov5-5.0-t2-onnx/runs/train/exp2/weights/best.onnx";
    //string model_path = "E:/IDM_download/yolov5s_2.onnx";
    Yolo test;
    Net net;
    if (test.readModel(net, model_path, true)) {
        cout << "read net ok!" << endl;
    }
    //���������ɫ
    vector<Scalar> color;
    srand(time(0));
    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(Scalar(b, g, r));
    }
    vector<Output> result;
    Mat img = imread(img_path);
    if (test.Detect(img, net, result)) {
        //��YOLOv5����ѵ������ͼƬ����
        test.drawPred(img, result, color);

    }
    else {
        cout << "Detect Failed!" << endl;
    }
}

