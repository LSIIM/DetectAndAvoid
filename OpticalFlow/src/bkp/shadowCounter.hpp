#ifndef _SHADOW_COUNTER_H
#define	_SHADOW_COUNTER_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

#include "background.hpp"

using std::vector;

namespace bammPoC{
  
class ShadowCounter
{
     
public:
      //constructor and destructor methods
      ShadowCounter();
      ~ShadowCounter(); 
      
      void Execute();
      void SetImage(cv::Mat& inputImage);
      cv::Point GetSeedPoint() const { return m_pxy; };

private:
      //input image dimensions and pointer
      unsigned int m_W, m_H;
      cv::Mat m_inputImage;    
      //background by a region growing method
      cv::Mat m_background;
      
      //static random colors to discriminate cells
      cv::Scalar m_rancolors[128];      
      
      //mouse click -- selection
      cv::Point m_pxy;
      //using this variable we can avoid unnecessary processing (e.g.background)
      bool m_newpos;
      
      //algorithm parameters
      int m_hoDp, m_hoMinDeRa, m_holowTh, m_hohighTh, m_hoMinRa, m_hoMaxRa;
      int m_cannylowThreshold, m_cannyhighThreshold, m_cannykernel_size;  
      int m_backgroundThreshold;
      
      float m_circle_proportion;
      float m_circle_excentrity;
      int m_circle_area_min, m_circle_area_max;
      
      //private methods
      //read canny parameters
      bool readCanny(const char* filename);
      //read hough parameters
      bool readHough(const char* filename);
      //read background parameters
      bool readBackground(const char* filename);
      //read circle detection parameters
      bool readCircleParameters(const char* filename);
      
      //getBackground ROI corresponding to an external polygon
      cv::Mat GetBackgroundROI(unsigned int y, unsigned int x);      
      cv::Mat GetBackgroundROIOpenCV(unsigned int y, unsigned int x);
      
      //getBackgroundMask, here m_background is filled
      cv::Mat GetRegionGrowingArea(unsigned int y, unsigned int x);      

      //get a blured+canny image
      cv::Mat GetGradientCanny(cv::Mat inputImage, const char* filename);
      
      //detect circles
      std::vector<std::vector<cv::Point> > DetectCircles(std::vector<std::vector<cv::Point> > contours);
      
      //compute the center of a contour list and ratio
      vector<cv::Vec3f> GetPolygonCenters(std::vector<std::vector<cv::Point> > contours);
      
      //dilate and erode
      cv::Mat Erode(const cv::Mat& image);
      cv::Mat Dilate(const cv::Mat& image);      
      
      //*** Detection methods ***\\
      //detection is performed by canny variants
      cv::Mat DetectByGradient(cv::Mat backgroundroi, vector<cv::Vec3f>& circles_grad);
      
      //detection is performed by hough circles
      cv::Mat DetectByHough(cv::Mat backgroundroi);
      
      //detection is performed by background subtraction methods
      cv::Mat DetectByBGS(cv::Mat backgroundroi, vector<cv::Vec3f>& circles_bgs);

      //detection is performed by background subtraction methods + canny
      cv::Mat DetectByBGSCanny(cv::Mat backgroundroi, vector<cv::Vec3f>& circles_bgscanny);
      
      //integrate results from three detectors
      cv::Mat IntegrateApproaches(vector<cv::Vec3f> m1, vector<cv::Vec3f> m2, vector<cv::Vec3f> m3);      
      
      //member variables used to viewers
      cv::Mat result1Gradient, result3BGS, result4BGSCanny;
      cv::Mat backgroundroi;
      cv::Mat resultMerged;
      vector<cv::Vec3f> circles_grad, circles_bgs, circles_bgscanny;
      
protected:
      static void mouseEvent(int evt, int x, int y, int flags, void* param);
};

};

#endif
    
    
