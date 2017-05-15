#include "tracker/tracker.h"
#include "network/regressor.h"
#include "helper/bounding_box.h"

#include <opencv/cv.h>

#include <iomanip>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>

using std::string;

class TrackStart {
    public:
    int x1, y1, x2, y2;
    string imname;
    TrackStart(string, int, int, int, int);
    
};

TrackStart::TrackStart(string i, int ix1, int iy1, int ix2, int iy2) {
    imname = i;
    x1 = ix1;
    y1 = iy1;
    x2 = ix2;
    y2 = iy2;
}

int main (int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel folder"
              << std::endl;
    return 1;
  }

  const string& model_file   = argv[1];
  const string& trained_file = argv[2];
  const string& image_folder = argv[3];
  
  const bool do_train = false;
  
  int gpu_id = -1;
  Regressor regressor(model_file, trained_file, gpu_id, do_train);
  
  srandom(time(NULL));
  
  const bool show_intermediate_output = false;
  Tracker tracker(show_intermediate_output);
  
  string images_path = image_folder + "/images.txt";
  std::ifstream images_file(images_path.c_str());
  std::vector<string> images;
  string line;
  while (std::getline(images_file, line)) {
    images.push_back(line);
  }
  images_file.close();
  
  std::cout << "According to images.txt, there are " << images.size() << " images." << std::endl;

  string boxes_path = image_folder + "/boxes.txt";
  std::ifstream boxes_file(boxes_path.c_str());
  std::vector<TrackStart> track_starts;
  while (std::getline(boxes_file, line)) {
    std::istringstream iss(line);
    string limname;
    int x1,y1,x2,y2;
    if (!(iss >> limname >> x1 >> y1 >> x2 >> y2)) { break; }
    TrackStart ts(limname, x1, y1, x2, y2);
    track_starts.push_back(ts);
  } 
  boxes_file.close();
  
  int inum = 0;
  for (std::vector<TrackStart>::iterator it = track_starts.begin(); it != track_starts.end(); ++it) {
    ++inum;
    std::ofstream out;
    std::ostringstream nss;
    nss << image_folder << "/out" << inum << ".txt";
    out.open(nss.str().c_str());
    
    std::vector<string>::iterator imit = images.begin();
    while (*imit != it->imname) ++imit;
    
    const cv::Mat& im_curr = cv::imread(image_folder + it->imname);
  
    std::vector<float> bbox_poss;
    bbox_poss.push_back(it->x1);
    bbox_poss.push_back(it->y1);
    bbox_poss.push_back(it->x2);
    bbox_poss.push_back(it->y2);
    BoundingBox bbox_gt(bbox_poss);

    tracker.Init(im_curr, bbox_gt, &regressor); 

    for (; imit != images.end(); ++imit) {

        const string path = image_folder + "/" + *imit;
        const cv::Mat& image = cv::imread(path);
        BoundingBox bbox;
        tracker.Track(image, &regressor, &bbox);
        std::cout << path << std::endl;
        out << path << std::endl;
        out << bbox.x1_ << " " << bbox.y1_ << " " << bbox.x2_ << " " << bbox.y2_ << std::endl;
    }
    out.close();
  }
  
  std::cout << "huge success" << std::endl;
}
