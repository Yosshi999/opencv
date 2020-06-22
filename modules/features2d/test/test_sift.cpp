// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

//#define GENERATE_DATA // generate data in debug mode

namespace opencv_test { namespace {

TEST(Features2d_SIFT, regression_exact)
{
    Mat image = imread(cvtest::findDataFile("features2d/tsukuba.png"));
    string xml = cvtest::findDataFile("sift/result.xml.gz");

    ASSERT_FALSE(image.empty());

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    vector<KeyPoint> calcKeypoints;
    Mat calcDescriptors;
    Ptr<SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(gray, Mat(), calcKeypoints, calcDescriptors, false);

#ifdef GENERATE_DATA
    // create XML
    FileStorage fs(xml, FileStorage::WRITE);
    ASSERT_TRUE(fs.isOpened()) << xml;

    std::cout << "Creating xml..." << std::endl;
    fs << "keypoints" << calcKeypoints;
    fs << "descriptors" << calcDescriptors;
    fs.release();
#else
    // read XML
    FileStorage fs(xml, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened()) << xml;

    vector<KeyPoint> validKeypoints;
    Mat validDescriptors;
    read( fs["keypoints"], validKeypoints );
    read( fs["descriptors"], validDescriptors, Mat() );
    fs.release();

    // Compare the number of keypoints
    ASSERT_EQ(validKeypoints.size(), calcKeypoints.size())
        << "Bad keypoints count (validCount = " << validKeypoints.size()
        << ", calcCount = " << calcKeypoints.size() << ")";

    // Compare the order and coordinates of keypoints
    size_t exactCount = 0;
    for (size_t i = 0; i < validKeypoints.size(); i++) {
        float dist = (float)cv::norm( calcKeypoints[i].pt - validKeypoints[i].pt );
        if (dist == 0) {
            exactCount++;
        }
    }
    ASSERT_EQ(exactCount, validKeypoints.size())
        << "Keypoints mismatch: exact count (dist==0) is " << exactCount << "/" << validKeypoints.size() << ".";

    // Compare descriptors
    ASSERT_EQ(validDescriptors.size, calcDescriptors.size) << "Size of descriptors mismatch.";
    ASSERT_EQ(0, cvtest::norm(validDescriptors, calcDescriptors, NORM_L2)) << "Descriptors mismatch.";
#endif // GENERATE_DATA
}

}} // namespace
