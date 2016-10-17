
#include <stdio.h>
//#include <iterator>
//#include <algorithm>
//#include <glog/logging.h>
#include "opencv2/opencv.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
//#include "classifier.h"
//#include "face_align.h"
#include "verification.h"

//namespace fs = ::boost::filesystem;

namespace face_ver_srzn {
using namespace cv;
using namespace std;

Verification::Verification(const string &model_file, const string &trained_file, const string & feature_name, bool use_gpu)
    :_classifier(model_file, trained_file, string(), use_gpu)
{
    _input_size = _classifier.GetInputGeometry();
    _input_channels = _classifier.GetInputChannels();
    _feature_name = feature_name;
    _feature_dim = _classifier.GetLayerDimByName(_feature_name);
    assert(_feature_dim > 0);
    assert(_input_channels == 1 || _input_channels == 3);
}

int Verification::GetFeatureDim() {
    return _feature_dim;
}

  void Verification::ExtractFeature(const Mat &face, Mat &feature){
    Mat img(face);
    if (face.channels() == 3)
        cvtColor(face, img, CV_BGR2GRAY);
    img.convertTo(img, CV_32FC1, 1.0/255);
    _classifier.ExtractLayerByName(img, _feature_name, feature);
  }

  Rect Verification::DetectFaceAndExtractFeature(const Mat &img, Mat &feature){
      Mat face;
      Rect face_rect = DetectFaceAndAlign(img, face);
      if (face_rect.area() > 0)
          ExtractFeature(face, feature);
      return face_rect;
  }

  float Verification::Compare(const Mat & img1, const Mat & img2){
      Mat f1, f2;
      ExtractFeature(img1, f1);
      ExtractFeature(img2, f2);
      return CompareFeature(f1, f2);
  }

  float Verification::CompareFeature(const Mat & feature_a, const Mat & feature_b){
      double ab = feature_a.dot(feature_b);
      double aa = feature_a.dot(feature_a);
      double bb = feature_b.dot(feature_b);
      return -ab / sqrt(aa*bb);;
  }

  Rect Verification::DetectFaceAndAlign(const Mat & img, Mat & face){
      return Rect();
  }
}
