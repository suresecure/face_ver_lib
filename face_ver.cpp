#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iostream>
//#include <sstream>
//#include <iterator>
//#include <algorithm>

#include "opencv2/opencv.hpp"

#include "face_align.h"
#include "verification.h"

using namespace cv;
using namespace std;
using namespace face_ver_srzn;

int readImageList(const string & listfile, vector<string> & list) {
    ifstream f(listfile);
    list.clear();
    string line;
    while (getline(f, line))
        list.push_back(line);
    return list.size();
}

long extractImageFeature(vector<string> & list, Mat & features, Verification & ver) {
    long time1 = clock();
    cout<<"Extracting image feature: "<<endl;
    for (int i = 0; i < list.size(); i++ ){
        Mat img = imread(list[i], CV_LOAD_IMAGE_GRAYSCALE);
        Mat feature;
        ver.ExtractFeature(img, feature);
        feature.copyTo(features.row(i));
        cout<<i+1<<"/"<<list.size()<<endl;
    }
    long time2 = clock();
    return time2-time1;
}

inline void dumpFeatureLine(ofstream &of, const Mat feature) {
    for (int i = 0; i < feature.cols; i++)
        of<<feature.at<float>(i)<<" ";
    of<<endl;
}

int main(int argc, char **argv) {

    string cnn_model_path = "/home/robert/myCoding/face/face_feature_cnn_models/face_verification_experiment/model/LightenedCNN_B.caffemodel";
    string cnn_proto_path = "/home/robert/myCoding/face/face_feature_cnn_models/face_verification_experiment/proto/LightenedCNN_B_deploy.prototxt";
    string feature_name = "eltwise_fc1";
    string align_model_path = "/home/robert/myCoding/suresecure/face_rec_models/shape_predictor_68_face_landmarks.dat";
    bool use_gpu = false;
    string image_path = "/home/robert/myCoding/suresecure/dataset/lfw_data/lfw_align_128x128_lightened_cnn/Aaron_Eckhart/Aaron_Eckhart_0001.png";
    string image_not_align_path = "/home/robert/myCoding/suresecure/dataset/test_faces/Jennifer_Renee_Short_0001.jpg";
    string image_list_path = string("/home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/align_wu_origin/");
    string image_list_name = string("aligned_id.txt");
//    string image_list_name = string("aligned_frontal_xcq.txt");
    image_list_path += image_list_name;

    /*
    // Test Classifier
    //Mat img = imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    //img.convertTo(img, CV_32FC1, 1.0/255);
    //Mat feature;
    //Classifier * conv_net = new Classifier( cnn_proto_path,  cnn_model_path);
    //conv_net->ExtractLayerByName(img, feature_name, feature);
    //cout<<feature<<endl;
    // Test Verification
    Mat img = imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    Mat feature;
    Verification ver(cnn_proto_path, cnn_model_path, feature_name, use_gpu);
    ver.ExtractFeature(img, feature);
    cout<<feature<<endl;
    */

    /*
    // Test feature extraction of Verification.
    Verification ver(cnn_proto_path, cnn_model_path, feature_name, use_gpu);
    vector<string> image_list;
    int num_images = readImageList(image_list_path, image_list);
    Mat features(num_images, ver.GetFeatureDim(), CV_32FC1);
    extractImageFeature(image_list, features,  ver);
    // Dump feature to txt file
    ofstream of(image_list_name);
    for (int i = 0; i < num_images; i++ ) {
        string fname = image_list[i];
//        int pos = fname.find_last_of("/\\")+1;
//        of<<fname.substr(pos, fname.find_last_of('.')-pos)<<" ";
        of<<fname.substr(fname.find_last_of("/\\")+1)<<" ";
        dumpFeatureLine(of, features.row(i));
    }
    */

    
    /*
    // Test face align.
    FaceAlign align(align_model_path);
    Mat img = imread(image_not_align_path);
    Mat face;
    Rect face_rect = align.detectAlignCropLigntenedCNNOrigin(img, face);
    if (face_rect.area() > 0) {
        imshow("face", face);
        waitKey(-1); 
    }*/

    // Test face align for ID image.
    FaceAlign align(align_model_path);
    Mat img = imread("/home/robert/myCoding/suresecure/dataset/IDSamplePhoto/origin/id/430681199412212619-0.jpg");
    Mat face;
    Rect ave_face_rect(13.18, 31.42, 71.88, 71.88);
    Rect face_rect = align.detectAlignCropLigntenedCNNOrigin(img, face);
    if (face_rect.area() <= 0) {
        // Dlib face detector cannot detect face. Use average face rect instead.
        face_rect = align.detectAlignCropLigntenedCNNOrigin(img, face, 128, 48, 40, ave_face_rect);
    }
    imshow("face", face);
    waitKey(-1); 

    /*
    // Align and save face.
    string to_be_align_image_list_path = string("/home/robert/myCoding/suresecure/dataset/IDSamplePhoto/filename.txt");
    vector<string> image_list;
    int num_images = readImageList(to_be_align_image_list_path, image_list);
    FaceAlign align(align_model_path);
    for (int i = 0; i < num_images; i++ ) {
        string fname = image_list[i];
        Mat img = imread(fname);
        Mat face;
        //// For training
        //Rect face_rect = align.detectAlignCropLigntenedCNNOrigin(img, face, 144, 48, 48);
        // For testing
        Rect face_rect = align.detectAlignCropLigntenedCNNOrigin(img, face);
        if (face_rect.area() <= 0 && fname.find("/id/") >= 0 ){
            // ID image
            // Dlib face detector cannot detect face. Use average face rect instead.
            Rect ave_face_rect(13.18, 31.42, 71.88, 71.88);
            face_rect = align.detectAlignCropLigntenedCNNOrigin(img, face, 128, 48, 40, ave_face_rect);
        }
        if (face_rect.area() <= 0){
            remove(fname.c_str());
            //cerr<<"ERROR: DETECT FAIL: "<<fname<<endl;
            cerr<<fname<<endl;
        }
        else {
            imwrite(fname, face);
            cout<<fname<<endl;
        }
    }*/

    /*
    // Average rect of ID images.
    string to_be_align_image_list_path = string("/home/robert/myCoding/suresecure/dataset/IDSamplePhoto/id.txt");
    vector<string> image_list;
    int num_images = readImageList(to_be_align_image_list_path, image_list);
    FaceAlign align(align_model_path);
    float x1=0, x2=0, y1=0, y2=0;
    for (int i = 0; i < num_images; i++ ) {
        string fname = image_list[i];
        Mat img = imread(fname);
        Mat face;
        // For testing
        Rect face_rect = align.detectAlignCropLigntenedCNNOrigin(img, face);
        if (face_rect.area() <= 0){
            cerr<<"ERROR: DETECT FAIL: "<<fname<<endl;
        }
        else {
            x1 += face_rect.x;
            x2 += face_rect.width;
            y1 += face_rect.y;
            y2 += face_rect.height;
            cout<<fname<<endl;
        }
    }
    x1 /= num_images;
    x2 /= num_images;
    y1 /= num_images;
    y2 /= num_images;
    cout<<"Average face rect of ID images: "<<x1<<", "<<x2<<", "<<y1<<", "<<y2<<endl;
    */

    return 0;
}
