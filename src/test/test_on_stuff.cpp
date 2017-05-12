#include "tracker/tracker.h"
#include "network/regressor.h"
#include "helper/bounding_box.h"

#include <opencv/cv.h>

using std::string;

int main (int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << std::endl;
    return 1;
  }

  const string& model_file   = argv[1];
  const string& trained_file = argv[2];
  const string& image_folder = "/tracking/data/vid1/";
  
  const bool do_train = false;
  
  int gpu_id = -1;
  Regressor regressor(model_file, trained_file, gpu_id, do_train);
  
  srandom(time(NULL));
  
  const bool show_intermediate_output = false;
  Tracker tracker(show_intermediate_output);
  
  const cv::Mat& im_curr = cv::imread(image_folder + "/001.jpg");
  
  std::vector<float> bbox_poss;
  bbox_poss.push_back(233.0f);
  bbox_poss.push_back(466.0f);
  bbox_poss.push_back(312.0f);
  bbox_poss.push_back(630.0f);
  BoundingBox bbox_gt(bbox_poss);
  
  tracker.Init(im_curr, bbox_gt, &regressor); 
  
  std::cout << "huge success" << std::endl;
}
