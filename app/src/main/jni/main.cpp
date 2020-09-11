#include <jni.h>
#include <opencv2/opencv.hpp>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/imgproc/imgproc.hpp"


#include <android/log.h>

using namespace cv;
using namespace std;

// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 100;

cv::Mat imgNumbers;         // input image
cv::Mat imgGrayscale;       //
cv::Mat imgBlurred;         // declare various images
cv::Mat imgThresh;          //
cv::Mat imgThreshCopy;      //
cv::Rect ROI = Rect(600, 50, 500, 200);
cv::Rect ROI_Aim = Rect(595, 45, 510, 210);

std::vector<std::vector<cv::Point> > ptContours;        // declare contours vector
std::vector<cv::Vec4i> v4iHierarchy;                    // declare contours hierarchy

vector<vector<int>> arrDigit = {
          {1, 1, 1, 0, 1, 1, 1}       //0
        , {0, 0, 1, 0, 0, 1, 0}     //1
        , {1, 0, 1, 1, 1, 0, 1}     //2
        , {1, 0, 1, 1, 0, 1, 1}     //3
        , {0, 1, 1, 1, 0, 1, 0}     //4
        , {1, 1, 0, 1, 0, 1, 1}     //5
        , {1, 1, 0, 1, 1, 1, 1}     //6
        , {1, 1, 1, 0, 0, 1, 0}     //7
        , {1, 1, 1, 1, 1, 1, 1}     //8
        , {1, 1, 1, 1, 0, 1, 1}     //9
};
///////////////////////////////////////////////////////////////////////////////////////////////////

void DetectNumber(Mat &frame);

extern "C"
JNIEXPORT void JNICALL
Java_com_example_opencvandroid_MainActivity_DetectNumber(JNIEnv *env, jobject thiz,
        jlong mat_addr_input,
jlong mat_addr_result) {
Mat &matInput = *(Mat *)mat_addr_input;
Mat mat_gray, mat_detected_edges;
Mat &matResult = *(Mat *)mat_addr_result;
matResult = matInput.clone();

DetectNumber(matResult);
}

void DetectNumber(Mat &frame) {

    cv::rectangle(frame, ROI_Aim, cv::Scalar(0, 0, 255), 2);

    imgNumbers = Mat(frame, ROI);

    cv::cvtColor(imgNumbers, imgGrayscale, cv::COLOR_BGR2GRAY);        // convert to grayscale

    cv::GaussianBlur(imgGrayscale,              // input image
                     imgBlurred,                             // output image
                     cv::Size(5, 5),                         // smoothing window width and height in pixels
                     0);                                     // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

    // filter image from grayscale to black and white
    cv::adaptiveThreshold(imgBlurred,           // input image
                          imgThresh,                              // output image
                          255,                                    // make pixels that pass the threshold full white
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,         // use gaussian rather than mean, seems to give better results
                          cv::THRESH_BINARY_INV,                  // invert so foreground will be white, background will be black
                          11,                                     // size of a pixel neighborhood used to calculate threshold value
                          2);                                     // constant subtracted from the mean or weighted mean

    imgThreshCopy = imgThresh.clone();          // make a copy of the thresh image, this in necessary b/c findContours modifies the image

    cv::findContours(imgThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
                     ptContours,                             // output contours
                     v4iHierarchy,                           // output hierarchy
                     cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
                     cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points

    // ======================== 이승연 추가 ========================
    map<int, vector<vector<Point>>> list;
    map<int, vector<vector<Point>>> digitList;
    map<int, int> digitList2;                   //
    vector<vector<Point>> digitContours;

    // =================== 세로 그룹 기준 찾기 ===================
    int top = 0;
    int middle = 0;
    int bottom = 0;
    int gap = 9;

    for (int i = 0; i < ptContours.size(); i++)                                     // for each contour
    {
        if (contourArea(ptContours[i]) > MIN_CONTOUR_AREA)                          // if contour is big enough to consider
        {
            cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                // get the bounding rect

            if (boundingRect.height < gap + 10)
            {
                if (bottom == 0 || boundingRect.y + gap >= bottom)
                {
                    bottom = boundingRect.y;
                }
                else if (middle == 0 || boundingRect.y + gap >= middle)
                {
                    middle = boundingRect.y;
                }
                else if (top == 0 || boundingRect.y <= top + gap)
                {
                    top = boundingRect.y;
                }
            }
        }
    }
    // =================== 세로 그룹 기준 찾기 ===================

    int groupCnt = 0;
    int stdWidth = 0;
    vector<Rect> listBottom;
    // =================== 가로 영역중 하단 기준으로 숫자영역 찾기 ===================
    for (int i = 0; i < ptContours.size(); i++)                                             // for each contour
    {
        if (contourArea(ptContours[i]) > MIN_CONTOUR_AREA)                                  // if contour is big enough to consider
        {
            cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                        // get the bounding rect
            cv::rectangle(imgNumbers, boundingRect, cv::Scalar(0, 255, 0), 2);              // draw red rectangle around each contour as we ask user for input

            // 가로 영역인것만
            if (boundingRect.height < gap + 10)
            {
                // 하단영역들 중에
                if (boundingRect.y < bottom + gap)
                {
                    if (listBottom.size() < 1)
                    {
                        listBottom.push_back(boundingRect);
                    }
                    else
                    {
                        for (int j = 0; j < listBottom.size(); j++)
                        {
                            int tmpX = abs(listBottom[j].x - boundingRect.x);

                            if (tmpX > gap)
                            {
                                listBottom.push_back(boundingRect);
                                stdWidth = boundingRect.width;
                            }//end if
                        }//end for
                    }//end if
                }//end if
            }//end if
        }//end if
    }//end for
    // =================== 가로 영역중 하단 기준 찾기 ===================

    // =================== 영역별 그룹핑 ===================
    for (int i = 0; i < ptContours.size(); i++)                                     // for each contour
    {
        bool isGroupping = false;
        if (contourArea(ptContours[i]) > MIN_CONTOUR_AREA)                          // if contour is big enough to consider
        {
            cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                // get the bounding rect

            // 그룹별로 나누기
            for (int j = 0; j < listBottom.size(); j++)
            {
                // 기준위치안에 들어오는지 확인 좌측기준보다 크고, 우측(x좌표 + 너비)보다 작아야한다.
                if (listBottom[j].x - gap < boundingRect.x && listBottom[j].x + gap + listBottom[j].width > boundingRect.x)
                {
                    list[j].push_back(ptContours[i]);
                    isGroupping = true;
                    break;
                }//end if
            }//end for

            //1은 넘버링이 안됨
            if (!isGroupping)
            {
                list[listBottom.size()].push_back(ptContours[i]);
            }//end if
        }//end if
    }//end for
    // =================== 영역별 그룹핑 ===================

    // =================== 그룹별 영역지정 ===================
    for (auto p : list) {
        int minX = 0;
        int minY = 0;
        int maxX = 0;
        int maxY = 0;

        int width = 0;
        int height = 0;

        for (vector<Point> v : p.second) {
            Rect boundingRect1 = cv::boundingRect(v);

            if (width == 0 || boundingRect1.width > width)
            {
                width = boundingRect1.width;
            }//end if

            if (height == 0 || boundingRect1.height > height)
            {
                height = boundingRect1.height;
            }//end if

            if (minX == 0 || boundingRect1.x < minX)
            {
                minX = boundingRect1.x;
            }//end if

            if (minY == 0 || boundingRect1.y < minY)
            {
                minY = boundingRect1.y;
            }//end if

            if (maxX == 0 || boundingRect1.x + boundingRect1.width > maxX)
            {
                maxX = boundingRect1.x + boundingRect1.width;
            }//end if

            if (maxY == 0 || boundingRect1.y + boundingRect1.height > maxY)
            {
                maxY = boundingRect1.y + boundingRect1.height;
            }//end if
        }//end for

        //1일경우
        if (width < stdWidth - gap)
        {
            minX -= stdWidth - gap;

            if (minX < 0)
            {
                minX = 0;
            }//end if
        }//end if

        digitContours.push_back({ Point(minX, minY), Point(minX, maxY) , Point(maxX, minY) , Point(maxX, maxY) });
    }//end for
    // =================== 그룹별 영역지정 ===================

    // =================== 영역별 segment확인하여 숫자인식 ===================
    for (auto p : list) {
        vector<int> chkSeg{ 0,0,0,0,0,0,0 };
        int stdX = 0;   //기준값
        int stdY = 0;   //기준값

        if (p.first == listBottom.size())
        {
            for (vector<Point> v : p.second)
            {
                Rect boundingRect = cv::boundingRect(v);
                if (stdX == 0 || boundingRect.x - stdWidth < stdX)
                {
                    stdX = boundingRect.x - stdWidth;
                }//end if

                if (stdY == 0 || boundingRect.y < stdY)
                {
                    stdY = boundingRect.y;
                }//end if
            }//end for
        }
        else
        {
            for (vector<Point> v : p.second) {
                Rect boundingRect = cv::boundingRect(v);
                if (stdX == 0 || boundingRect.x < stdX)
                {
                    stdX = boundingRect.x;
                }//end if

                if (stdY == 0 || boundingRect.y < stdY)
                {
                    stdY = boundingRect.y;
                }//end if
            }//end for
        }//end if

        for (vector<Point> v : p.second) {
            Rect boundingRect = cv::boundingRect(v);

            //가로 Segment
            if (boundingRect.height < gap + 10)
            {
                int tmpY = abs(boundingRect.y - top);

                //1segment chekck
                if (tmpY < gap)
                {
                    chkSeg[0] = 1;
                }//end if

                tmpY = abs(boundingRect.y - middle);

                //4segment chekck
                if (tmpY < gap)
                {
                    chkSeg[3] = 1;
                }//end if

                tmpY = abs(boundingRect.y - bottom);

                //7segment chekck
                if (tmpY < gap)
                {
                    chkSeg[6] = 1;
                }//end if
            }
                //세로 Segment
            else
            {
                int tmpY = abs(boundingRect.y - top);
                int tmpX = abs(boundingRect.x - stdX);

                if (tmpY < gap)
                {
                    //2segment chekck
                    if (tmpX < gap)
                    {
                        chkSeg[1] = 1;
                    }
                        //3segment chekck
                    else
                    {
                        chkSeg[2] = 1;
                    }//end if
                }
                else
                {
                    //5segment chekck
                    if (tmpX < gap)
                    {
                        chkSeg[4] = 1;
                    }
                        //6segment chekck
                    else
                    {
                        chkSeg[5] = 1;
                    }//end if
                }//end if
            }//end if
        }//end if

        // 숫자체크
        int digit = find(arrDigit.begin(), arrDigit.end(), chkSeg) - arrDigit.begin();
        std::cout << "=============" << endl;
        std::cout << digit << endl;

        if (digit < 10)
        {
            digitList[digit] = p.second;
            digitList2[stdX] = digit;
            //cv::putText(imgNumbers, to_string(digit), Point(stdX, stdY), 1, 3, Scalar(0, 0, 0));
        }
        else
        {
            std::cout << digit << endl;
        }//end if

    }//end for
    // =================== 영역별 segment확인하여 숫자인식 ===================

    // ======================== 이승연 추가 ========================

    for (int i = 0; i < digitContours.size(); i++) {                                                // for each contour
        cv::Rect boundingRect1 = cv::boundingRect(digitContours[i]);                                // get the bounding rect

        cv::rectangle(imgNumbers, boundingRect1, cv::Scalar(255, 0, 0), 2);                 // draw red rectangle around each contour as we ask user for input

        //cv::Mat matROI1 = imgThresh(boundingRect1);           // get ROI image of bounding rect

        //cv::Mat matROIResized1;
        //cv::resize(matROI1, matROIResized1, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));   // resize image, this will be more consistent for recognition and storage

    }// end for

    int x = 800;
    int y = 250;
    for (auto const& entry : digitList2)
    {
        cv::putText(frame, to_string(entry.second), Point(x, y), 1, 3, Scalar(0, 0, 0));
        x += 25;
    }

    imgNumbers.copyTo(frame(ROI));
    //cv::imshow("imgNumbers", frame);                                      // show training numbers image, this will now have red rectangles drawn on it
}

extern "C"
JNIEXPORT jlong JNICALL
        Java_com_example_opencvandroid_MainActivity_loadCascade(JNIEnv *env, jobject thiz, jstring cascade_file_name) {
const char *nativeFileNameString = env->GetStringUTFChars(cascade_file_name, 0);

string baseDir("/storage/emulated/0/");
baseDir.append(nativeFileNameString);
const char *pathDir = baseDir.c_str();

jlong ret = 0;
ret = (jlong) new CascadeClassifier(pathDir);

if (((CascadeClassifier *) ret)->empty()) {
__android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ","CascadeClassifier로 로딩 실패  %s", nativeFileNameString);
}
else
__android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ","CascadeClassifier로 로딩 성공 %s", nativeFileNameString);

env->ReleaseStringUTFChars(cascade_file_name, nativeFileNameString);

return ret;
}
