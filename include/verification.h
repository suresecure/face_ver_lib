#ifndef VERIFICATION_H
#define VERIFICATION_H
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "classifier.h"

namespace face_ver_srzn {
using namespace cv;
using namespace std;

class Verification {
public:
  Verification(const string &model_file, const string &trained_file,
               const string & feature_name = "eltwise_fc1", bool use_gpu = false);

  void ExtractFeature(const Mat &face, Mat &feature);
  Rect DetectFaceAndExtractFeature(const Mat &img, Mat &feature);
  float Compare(const Mat & img1, const Mat & img2);
  float CompareFeature(const Mat & f1, const Mat & f2);
  Rect DetectFaceAndAlign(const Mat & img, Mat & face);
  int GetFeatureDim();

private:
  Classifier _classifier;
  cv::Size _input_size;
  int _input_channels;
  int _feature_dim;
  string _feature_name;
};
}

#endif //VERIFICATION_H
#
