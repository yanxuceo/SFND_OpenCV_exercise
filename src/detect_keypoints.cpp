#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

/**
 *  Shi-Tomasi and FAST implementation and performance comparison
 */

void detKeypoints1()
{
    // load image from file and convert to grayscale
    cv::Mat imgGray;
    cv::Mat img = cv::imread("../images/img1.png");
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // Shi-Tomasi detector

    //  size of a block for computing a derivative covariation matrix over each pixel neighborhood
    int blockSize = 6;     

    // max. permissible overlap between two features in % 
    double maxOverlap = 0.0; 
    double minDistance = (1.0 - maxOverlap) * blockSize;
    
    // max. num. of keypoints
    int maxCorners = img.rows * img.cols / max(1.0, minDistance);
    // minimal accepted quality of image corners
    double qualityLevel = 0.01;                                 
   
    double k = 0.04;
    bool useHarris = false;

    vector<cv::KeyPoint> kptsShiTomasi;
    vector<cv::Point2f> corners;
    double t = (double)cv::getTickCount();
    cv::goodFeaturesToTrack(imgGray, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarris, k);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Shi-Tomasi with n= " << corners.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::abortendl;

    for (auto it = corners.begin(); it != corners.end(); ++it)
    { 
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        kptsShiTomasi.push_back(newKeyPoint);
    }

    // visualize results
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, kptsShiTomasi, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "Shi-Tomasi Results";
    cv::namedWindow(windowName, 1);
    imshow(windowName, visImage);

    
    // add the FAST detector and compare both algorithms with regard to 
    // (a) number of keypoints
    // (b) distribution of keypoints over the image
    // (c) processing speed
    
    // difference between intensity of the central pixel and pixels of a circle around this pixel
    int threshold = 30;

    // perform non-maxima suppression on keypoints
    bool bNMS = true; 

    // TYPE_9_16, TYPE_7_12, TYPE_5_8
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);

    vector<cv::KeyPoint> kptsFAST;
    t = (double)cv::getTickCount();
    detector->detect(imgGray, kptsFAST);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST with n= " << kptsFAST.size() << " keypoints in " << 1000*t/1.0 << " ms" << std::endl;

    visImage = img.clone();
    cv::drawKeypoints(img, kptsFAST, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    windowName = "FAST Results";
    cv::namedWindow(windowName, 2);
    imshow(windowName, visImage);

    cv::waitKey(0);
}

int main()
{
    detKeypoints1();
}