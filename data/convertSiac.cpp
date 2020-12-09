// This program converts a set of gray images to a leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
 
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "leveldb/db.h"
 
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include "google/protobuf/text_format.h"
#include "stdint.h"
#include <cstdio>
#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"
 
using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
 
DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");
DEFINE_int32(channel,3,"channel number of the image");
 
static bool ReadImageToMemory(const string& FileName, const int Height,
                              const int Width, char *Pixels)
{
    // read image
    cv::Mat OriginImage = cv::imread(FileName);
    CHECK(OriginImage.data) << "Failed to read the image.\n";
 
 
    // resize the image
    cv::Mat ResizeImage;
    cv::resize(OriginImage, ResizeImage, cv::Size(Width, Height));
    CHECK(ResizeImage.rows == Height) << "The heighs of Image is no equal to the input height.\n";
    CHECK(ResizeImage.cols == Width) << "The width of Image is no equal to the input width.\n";
    CHECK(ResizeImage.channels() == 3) << "The channel of Image is no equal to one.\n";
 
    LOG(INFO) << "The height of image is " << ResizeImage.rows << "\n";
    LOG(INFO) << "The width of image is " << ResizeImage.cols << "\n";
    LOG(INFO) << "The channels of image is " << ResizeImage.channels() << "\n";
 
    // copy the image data to Pixels
    for (int HeightIndex = 0; HeightIndex < Height; ++HeightIndex)
    {
        const uchar* ptr = ResizeImage.ptr<uchar>(HeightIndex);
        int img_index = 0;
        for (int WidthIndex = 0; WidthIndex < Width; ++WidthIndex)
        {
            for (int ChannelIndex = 0; ChannelIndex < ResizeImage.channels(); ++ChannelIndex)
            {
                int datum_index = (ChannelIndex * Height + HeightIndex) * Width + WidthIndex;
                *(Pixels + datum_index) = static_cast<char>(ptr[img_index++]);
            }
        }
    }
 
    return true;
}
 
 
int main(int argc, char** argv)
{
    //::google::InitGoogleLogging(argv[0]);
 
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
 
    gflags::SetUsageMessage("Convert a set of color images to the leveldb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
    caffe::GlobalInit(&argc, &argv);
 
    // 输入参数不足时报错
    if (argc < 4)
    {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convertSiac");
        return 1;
    }
 
 
    // 读取图像名字和标签
   /* std::ifstream infile(argv[2]);
    std::vector<std::pair<std::string,std::string> > lines;
    std::string filename;
    std::string pairname;
   // int label;
    while (infile>>filename>>pairname)
    {2205271
        lines.push_back(std::make_pair(filename, pairname));
    }*/
    std::ifstream infile(argv[2]);
    std::vector<std::string>line(3);
    std::vector<std::vector<std::string> >lines(209343,line);
    std::string filename;
    std::string pairname;
    std::string label;
    int i=0;
    while (infile>>filename>>pairname>>label)
    {
	
        lines[i][0]=filename;
	lines[i][1]=pairname;
	lines[i][2]=label;
	i++;
    }
    // 打乱图片顺序
    if (FLAGS_shuffle)
    {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        shuffle(lines.begin(), lines.end());
    }
    LOG(INFO) << "A total of " << lines.size() << " images.";
 
 
 
    // 设置图像的高度和宽度
    int resize_height = std::max<int>(0, FLAGS_resize_height);
    int resize_width = std::max<int>(0, FLAGS_resize_width);
    int channel=std::max<int>(1,FLAGS_channel);
 
 
    // 打开数据库
    // Open leveldb
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    leveldb::Status status = leveldb::DB::Open(
        options, argv[3], &db);
    CHECK(status.ok()) << "Failed to open leveldb " << argv[3]
        << ". Is it already existing?";
 
 
    // 保存到leveldb
    // Storing to leveldb
    std::string root_folder(argv[1]);
    char* Pixels = new char[2 * resize_height * resize_width*channel];
    const int kMaxKeyLength = 10;
    char key[kMaxKeyLength];
    std::string value;
 
    caffe::Datum datum;
    datum.set_channels(2*channel);  // one channel for each image in the pair
    datum.set_height(resize_height);
    datum.set_width(resize_width);
   // print("anc");
    //
    for (int LineIndex = 0; LineIndex < lines.size(); LineIndex++)
    {
       // int PairIndex = caffe::caffe_rng_rand() % lines.size();
 	//std::cout << lines[LineIndex][0]<<std::endl;
        char* FirstImagePixel = Pixels;
        ReadImageToMemory(root_folder + lines[LineIndex][0], resize_height, resize_width, FirstImagePixel);
 
        char *SecondImagePixel = Pixels + resize_width * resize_height*channel;
        ReadImageToMemory(root_folder + lines[LineIndex][1], resize_height, resize_width, SecondImagePixel);
 
        // set image pair data
        datum.set_data(Pixels, 2 * resize_height * resize_width*channel);
 
        // set label
        /*if (lines[LineIndex].second == lines[PairIndex].second)
        {
            datum.set_label(1);
        }
        else
        {
            datum.set_label(0);
        }*/
	datum.set_label(atoi(lines[LineIndex][2].c_str()));
        // serialize datum to string
        datum.SerializeToString(&value);
        int key_value = (int)(LineIndex);
        snprintf(key, kMaxKeyLength, "%08d", key_value);
        string keystr(key);
        std::cout << "label: " << datum.label() << ' ' << "key index: " << keystr << std::endl;
 
        db->Put(leveldb::WriteOptions(), std::string(key), value);
    }
 
    delete db;
    delete[] Pixels;
 
    return 0;
}
