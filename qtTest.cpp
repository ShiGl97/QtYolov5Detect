
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
    //注册文件格式
    GDALAllRegister();
    GDALDataset* poDataset;
    QString filename;
    filename = QFileDialog::getOpenFileName(this,
        tr("选择图像"),
        "",
        tr("Images (*.png *.bmp *.jpg *.tif *.GIF )"));

    //Qstring类型转换为const char*

    std::string str = filename.toStdString();
    const char* pszFileName = str.c_str();
    //const char* pszFileName = "E:\\OpenCV_image\\d_SRC_00_00.tif";
    //用只读方式打开图像
    poDataset = (GDALDataset*)GDALOpen(pszFileName, GA_ReadOnly);
    if (poDataset == NULL) {
        printf("打开失败!");
    }
    else {
        printf("打开成功!\n");

    }
    //获取图像格式信息
    std::cout << poDataset->GetDriver()->GetDescription() << std::endl;
    imageInfo1 = poDataset->GetDriver()->GetDescription();
    //输出图像的大小和波段个数
    printf("图像大小：%dx%dx\n",
        poDataset->GetRasterXSize(), poDataset->GetRasterYSize());
    imageInfo2 = QString::number(poDataset->GetRasterXSize(), 10);
    imageInfo2.append("*");
    imageInfo2.append(QString::number(poDataset->GetRasterYSize(), 10));
    printf("波段个数：%d\n", poDataset->GetRasterCount());
    imageInfo3 = QString::number(poDataset->GetRasterCount());
    //输出图像投影信息
    if (poDataset->GetProjectionRef() != NULL) {
        printf("投影信息:'%s'\n", poDataset->GetProjectionRef());
    }
    //输出图形的坐标和分辨率信息
    double adfGeoTransform[6];
    if (poDataset->GetGeoTransform(adfGeoTransform) == CE_None) {
        printf("坐标信息 = (%.6f,%.6f)\n", adfGeoTransform[0], adfGeoTransform[3]);
        printf("分辨率信息 = (%.6f,%.6f)\n", adfGeoTransform[1], adfGeoTransform[5]);

    }
    imageInfo4 = QString("%1").arg(adfGeoTransform[0]);
    imageInfo4.append(",");
    imageInfo4.append(QString("%1").arg(adfGeoTransform[3]));
    imageInfo5 = QString("%1").arg(adfGeoTransform[1]);
    imageInfo5.append(",");
    imageInfo5.append(QString("%1").arg(adfGeoTransform[5]));
    //imageInfo4 = adfGeoTransform[0] + adfGeoTransform[3];
    //imageInfo5 = adfGeoTransform[1] + adfGeoTransform[5];
    //获取栅格波段
    GDALRasterBand* poBand;
    int nBlockXSize, nBlockYSize;
    int nGotMin, nGotMax;
    double adfMinMax[2];
    //读取第一个波段
    poBand = poDataset->GetRasterBand(1);
    //获取该波段的最大最小值，如果获取失败，则进行统计
    adfMinMax[0] = poBand->GetMinimum(&nGotMin);
    adfMinMax[1] = poBand->GetMaximum(&nGotMax);
    if (!(nGotMin && nGotMax)) {
        GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE, adfMinMax);
    }
    printf("波段最小值 = %.3fd, 波段最大值 = %.3f\n", adfMinMax[0], adfMinMax[1]);

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
        tr("选择图像"),
        "",
        tr("Images (*.png *.bmp *.jpg *.tif *.GIF )"));
    if (filename.isEmpty())
    {
        return;
    }
    else
    {
        //QImage* img = new QImage;

        //if (!(img->load(filename))) //加载图像
        //{
        //	QMessageBox::information(this,
        //		tr("打开图像失败"),
        //		tr("打开图像失败!"));
        //	delete img;
        //	return;
        //}

        string str = filename.toStdString();  // 将filename转变为string类型；
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
        tr("选择图像"),
        "",
        tr("Images (*.png *.bmp *.jpg *.tif *.GIF )"));
    if (filename.isEmpty())
    {
        return;
    }
    else
    {
        //QImage* img = new QImage;

        //if (!(img->load(filename))) //加载图像
        //{
        //	QMessageBox::information(this,
        //		tr("打开图像失败"),
        //		tr("打开图像失败!"));
        //	delete img;
        //	return;
        //}

        string str = filename.toStdString();  // 将filename转变为string类型；
        Mat image = imread(str);
        //image=imread(fileName.toLatin1().data);
        cvtColor(image, image, COLOR_BGR2RGB);
        cv::resize(image, image, Size(380, 490));
        QImage img = QImage((const unsigned char*)(image.data), image.cols, image.rows, QImage::Format_RGB888);


        label = new QLabel();
        label->setPixmap(QPixmap::fromImage(img));
        label->resize(QSize(img.width(), img.height()));
        ui.scrollArea->setWidget(label);


        //flip(image, image, 4);//反转函数 0 上下反转；整数，水平发转；负数，水平垂直均反转
        GaussianBlur(image, image, Size(5, 5), 15);//高斯模糊
        //bilateralFilter(image, image, 0, 100, 10);//双边模糊
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
    blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0, 0), true, false);//如果训练集未对图片进行减去均值操作，则需要设置为这句
    //blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(114, 114,114), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> netOutputImg;
    //vector<string> outputLayerName{"345","403", "461","output" };
    //net.forward(netOutputImg, outputLayerName[3]); //获取output的输出
    net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
    std::vector<int> classIds;//结果id数组
    std::vector<float> confidences;//结果每个id对应置信度数组
    std::vector<cv::Rect> boxes;//每个id矩形框
    float ratio_h = (float)netInputImg.rows / netHeight;
    float ratio_w = (float)netInputImg.cols / netWidth;
    int net_width = className.size() + 5;  //输出的网络宽度是类别数+5
    float* pdata = (float*)netOutputImg[0].data;
    for (int stride = 0; stride < 3; stride++) {    //stride
        int grid_x = (int)(netWidth / netStride[stride]);
        int grid_y = (int)(netHeight / netStride[stride]);
        for (int anchor = 0; anchor < 3; anchor++) { //anchors
            const float anchor_w = netAnchors[stride][anchor * 2];
            const float anchor_h = netAnchors[stride][anchor * 2 + 1];
            for (int i = 0; i < grid_y; i++) {
                for (int j = 0; j < grid_y; j++) {
                    float box_score = Sigmoid(pdata[4]);//获取每一行的box框中含有某个物体的概率
                    if (box_score > boxThreshold) {
                        //为了使用minMaxLoc(),将85长度数组变成Mat对象
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
                    pdata += net_width;//指针移到下一行
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
        //置信度表示
        string label = className[result[i].id] + ":" + to_string(result[i].confidence);

        int baseLine;
        //Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
        putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
    }
    imshow("res", img);
    //实现选择路径的保存
    cout << "请输入保存路径:" << endl;
    string str;
    cin >> str;
    imwrite(str, img);
    waitKey();
    //destroyAllWindows();
}


void qtTest::on_CnnFig_clicked() {
    QString filename;
    filename = QFileDialog::getOpenFileName(this,
        tr("选择图像"),
        "",
        tr("Images (*.png *.bmp *.jpg *.tif *.GIF )"));

    //Qstring类型转换为const char*

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
    //生成随机颜色
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
        //经YOLOv5网络训练后结果图片保存
        test.drawPred(img, result, color);

    }
    else {
        cout << "Detect Failed!" << endl;
    }
}

