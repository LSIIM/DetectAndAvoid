#include "shadowCounter.hpp"
#include <sstream>
#include <stdio.h>

#define DEBUG 1
//-------------------------------------------------------------------------------------
void bammPoC::ShadowCounter::mouseEvent(int evt, int x, int y, int flags, void* param)
{
    bammPoC::ShadowCounter* wnd = (bammPoC::ShadowCounter*) param;
    if(evt==CV_EVENT_LBUTTONDOWN)
    {
        wnd->m_pxy = cv::Point(x, y);
	wnd->m_newpos=true;
	return;
    }
}
//---------------------------------------------------------------------------------
bammPoC::ShadowCounter::ShadowCounter()
{
      //initializers
      m_H = m_W = 0;
      
      m_pxy.x = m_pxy.y = -1;      
      m_newpos = false;
      
      //initializing random colors
      for(int i=0; i<128; i++)
      {
	    //m_rancolors[i]=cv::Scalar(rand()%128,rand()%128,rand()%128);
	    m_rancolors[i]=cv::Scalar(0,0,255);
      }
}
//---------------------------------------------------------------------------------
bammPoC::ShadowCounter::~ShadowCounter()
{
}
//---------------------------------------------------------------------------------
bool bammPoC::ShadowCounter::readHough(const char* filename)
{
      FILE* pf = fopen(filename, "r");
      if(!pf) return false;
      
      fscanf(pf, "%d %d %d %d %d %d", &m_hoDp, &m_hoMinDeRa, &m_holowTh, &m_hohighTh, &m_hoMinRa, &m_hoMaxRa);
      
      fclose(pf);
      return true;
}
//---------------------------------------------------------------------------------
bool bammPoC::ShadowCounter::readCanny(const char* filename)
{
      FILE* pf = fopen(filename, "r");
      if(!pf) return false;
      
      fscanf(pf, "%d %d %d", &m_cannylowThreshold, &m_cannyhighThreshold, &m_cannykernel_size);
      
      fclose(pf);
      return true;
}
//---------------------------------------------------------------------------------
bool bammPoC::ShadowCounter::readBackground(const char* filename)
{
      FILE* pf = fopen(filename, "r");
      if(!pf) return false;
      
      fscanf(pf, "%d", &m_backgroundThreshold);
      
      fclose(pf);
      return true;
}
//---------------------------------------------------------------------------------
bool bammPoC::ShadowCounter::readCircleParameters(const char* filename)
{
      FILE* pf = fopen(filename, "r");
      if(!pf) return false;
      
      m_circle_proportion = m_circle_excentrity = m_circle_area_min = m_circle_area_max = 0;      
      fscanf(pf, "%f %f %d %d", &m_circle_proportion, &m_circle_excentrity, &m_circle_area_min, &m_circle_area_max);
      
      fclose(pf);
      return true;
}
//---------------------------------------------------------------------------------
void bammPoC::ShadowCounter::SetImage(cv::Mat& inputImage)
{
      //copy image dimensions
      m_H = inputImage.rows;
      m_W = inputImage.cols;
      
      //copy internal input image
      m_inputImage = inputImage.clone();
}
//---------------------------------------------------------------------------------
void bammPoC::ShadowCounter::Execute()
{
      if(!m_inputImage.data) return;
      
      //show input image and check if there is a ROI
      cv::namedWindow("inputimage", CV_WINDOW_KEEPRATIO);    
      cv::imshow("inputimage", m_inputImage);
    
      cvWaitKey(1);
      cv::setMouseCallback("inputimage", mouseEvent, this);
      if(m_pxy.x == m_pxy.y && m_pxy.x == -1) return;
      
      //drawing and show results      
      cv::namedWindow("Background ROI", CV_WINDOW_KEEPRATIO);  
      cv::namedWindow("Region Growing", CV_WINDOW_KEEPRATIO);      
      if(DEBUG) cv::namedWindow("Result1:DGB", CV_WINDOW_KEEPRATIO);      
      if(DEBUG) cv::namedWindow("Result3:DBGS", CV_WINDOW_KEEPRATIO);
      if(DEBUG) cv::namedWindow("Result4:DBGSCanny", CV_WINDOW_KEEPRATIO);
      cv::namedWindow("ResultMerge", CV_WINDOW_KEEPRATIO);
      
      static int prevbgth = 3;
      cv::createTrackbar("background level", "Region Growing", &prevbgth, 10);
      if(m_backgroundThreshold!=prevbgth)
      {
	  m_backgroundThreshold=prevbgth;
	  m_newpos = true;	  
      }
      
      //only process next instructions if we have a new position from the user
      //warning: parameters will not be loaded again
      if(m_newpos)
      {      
	  circles_grad.clear();
	  circles_bgs.clear();
	  circles_bgscanny.clear();
	
	  //we have a ROI, so get its contourns
	  //this->readBackground("../config/bg_threshold.par");
	  backgroundroi = this->GetBackgroundROIOpenCV(m_pxy.y, m_pxy.x);
	  //cv::Mat backgroundroi  = this->GetBackgroundROI(m_pxy.y, m_pxy.x);
	  
	  //DETECTION STARTS HERE !
	  //shadow counting by gradient detection
	  result1Gradient = this->DetectByGradient(backgroundroi, circles_grad);   
	  
	  //shadow counting by background subtraction methods	  
	  result3BGS = this->DetectByBGS(backgroundroi, circles_bgs);      
	  
	  //shadow counting by background subtraction methods	  
	  result4BGSCanny = this->DetectByBGSCanny(backgroundroi, circles_bgscanny);      
	  
	  //integrate three methods
	  resultMerged = this->IntegrateApproaches(circles_grad, circles_bgs, circles_bgscanny);
      }      
            
      cv::imshow("Background ROI", backgroundroi);
      cv::imshow("Region Growing", m_background);
      if(DEBUG) cv::imshow("Result1:DGB", result1Gradient);
      if(DEBUG) cv::imshow("Result3:DBGS", result3BGS);
      if(DEBUG) cv::imshow("Result4:DBGSCanny", result4BGSCanny);
      cv::imshow("ResultMerge", resultMerged);
      
      cvWaitKey(100);
}
//---------------------------------------------------------------------------------
cv::Mat bammPoC::ShadowCounter::IntegrateApproaches(vector<cv::Vec3f> m1, vector<cv::Vec3f> m2, vector<cv::Vec3f> m3)
{
      cv::Mat imgMerge = m_inputImage.clone();
      
      vector<cv::Vec3f> resultCircles;
      for(size_t i = 0; i < m1.size(); i++) resultCircles.push_back(m1.at(i));
      for(size_t i = 0; i < m2.size(); i++) resultCircles.push_back(m2.at(i));
      for(size_t i = 0; i < m3.size(); i++) resultCircles.push_back(m3.at(i));
      
      //if there is no circles detected do nothing
      if(!resultCircles.size()) return imgMerge;
      
      //N*N is the worst case
      //so I will use the instructions below for performance issues, ensured by kcachegrind software profile
      cv::Point center1, center2;
      int radius1, radius2;
      
      int i=0;
      while(i<resultCircles.size()-1)
      {
	  center1.x = resultCircles[i][0];
	  center1.y = resultCircles[i][1];
	  radius1 = resultCircles[i][2];

	  //for(size_t j = i+1; j < resultCircles.size(); j++)
	  int j=i+1;
	  while(j<resultCircles.size())
	  {
	      if(i==j) continue;
	    
	      center2.x = resultCircles[j][0];
	      center2.y = resultCircles[j][1];
	      radius2 = resultCircles[j][2];
	      
	      float de = sqrt((center1.x-center2.x)*(center1.x-center2.x)+(center1.y-center2.y)*(center1.y-center2.y));

	      if(de < (radius1+radius2)/2*1.15)
	      {
		  resultCircles[i][0]=(center1.x+center2.x)/2;
		  resultCircles[i][1]=(center1.y+center2.y)/2;
		  resultCircles[i][2]=(radius1+radius2)/2;
		  resultCircles.erase(resultCircles.begin()+j);
		  j=i;
	      }
	      j++;
	  }
	  i++;
      }      
      
      //drawing information
      for(size_t i = 0; i < resultCircles.size(); i++)
      {
	    cv::Point center(cvRound(resultCircles[i][0]), cvRound(resultCircles[i][1]));
	    int radius = cvRound(resultCircles[i][2]);
	    circle( imgMerge, center, 3, m_rancolors[i % 128], 1, 2, 0 );
      }
      
      //drawing number of detected objects
      if(resultCircles.size())
      {
	    std::stringstream sst; sst << "#" << resultCircles.size();
	    cv::putText(imgMerge, sst.str(), cv::Point(10,40), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, cv::Scalar::all(255), 1.5, 1.5);
      }      
            
      //drawing circles
      /*
      for(size_t i = 0; i < m1.size(); i++)
      {
	  cv::Point center(cvRound(m1[i][0]), cvRound(m1[i][1]));
	  int radius = cvRound(m1[i][2]);
	  circle( imgMerge, center, 3, cv::Scalar(0,0,255), 1, 2, 0 );
      }
      for(size_t i = 0; i < m2.size(); i++)
      {
	  cv::Point center(cvRound(m2[i][0]), cvRound(m2[i][1]));
	  int radius = cvRound(m2[i][2]);
	  circle( imgMerge, center, 3, cv::Scalar(0,255,0), 1, 2, 0 );
      }
      for(size_t i = 0; i < m3.size(); i++)
      {
	  cv::Point center(cvRound(m3[i][0]), cvRound(m3[i][1]));
	  int radius = cvRound(m3[i][2]);
	  circle( imgMerge, center, 3, cvAntonio, I have no problems with this. I am actually happy because I need to focus on my own work to graduate. I am here to help anytime, but if I am not need on Friday, I would rather not come. ::Scalar(255,0,0), 1, 2, 0 );
      }
      */
      return imgMerge;
}
//---------------------------------------------------------------------------------
cv::Mat bammPoC::ShadowCounter::DetectByGradient(cv::Mat backgroundroi, vector<cv::Vec3f>& circles_grad)
{    
      //get gradient image based on canny edge detector      
      cv::Mat gradient;
      m_inputImage.copyTo(gradient, backgroundroi);
      gradient = GetGradientCanny(gradient, "../config/dbg_canny.par");

      //extract shadows using a countor methods
      std::vector<std::vector<cv::Point> > contours;
      cv::findContours(gradient.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

      //polynomial curve approximation with closed vertices
      std::vector<std::vector<cv::Point> > contoursPolyApprox;
      for (size_t idx = 0; idx < contours.size(); idx++)
      {
	    std::vector<cv::Point> polyApprox;
	    cv::approxPolyDP(contours.at(idx), polyApprox, 0.3, true);
	    contoursPolyApprox.push_back(polyApprox);
      }
      
      //detect only circles
      std::vector<std::vector<cv::Point> > circles = this->DetectCircles(contoursPolyApprox);
      
      //draw detected circles      
      cv::Mat resultingbeads = m_inputImage.clone();
      for (size_t idx = 0; idx < circles.size(); idx++)
	    cv::drawContours(resultingbeads, circles, idx, m_rancolors[idx % 128]);
      if(circles.size()) 
      {
	    std::stringstream sst; sst << "#" << circles.size();
	    cv::putText(resultingbeads, sst.str(), cv::Point(100,200), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 5, cv::Scalar::all(255), 10, 8);
      }
      
      //get circle center and radius
      circles_grad = this->GetPolygonCenters(circles);

      //show images here
      if(DEBUG) cv::namedWindow("Gradient", CV_WINDOW_KEEPRATIO);  
      if(DEBUG) cv::imshow("Gradient", gradient);

      return resultingbeads;
}
//---------------------------------------------------------------------------------
cv::Mat bammPoC::ShadowCounter::DetectByHough(cv::Mat backgroundroi)
{           
      //get gradient image based on canny edge detector      
      cv::Mat gradient;
      m_inputImage.copyTo(gradient, backgroundroi);
      gradient = GetGradientCanny(gradient, "../config/dbh_canny.par");

      //extract shadows using a countor methods
      std::vector<std::vector<cv::Point> > contours;
      cv::findContours(gradient.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

      //polynomial curve approximation with closed vertices
      std::vector<std::vector<cv::Point> > contoursPolyApprox;
      for (size_t idx = 0; idx < contours.size(); idx++)
      {
	    std::vector<cv::Point> polyApprox;
	    cv::approxPolyDP(contours.at(idx), polyApprox, 0.3, true);
	    contoursPolyApprox.push_back(polyApprox);
      }
      
      //detect only circles
      std::vector<std::vector<cv::Point> > circles = this->DetectCircles(contoursPolyApprox);
      
      //draw detected circles     
      cv::Mat newgradient(cv::Mat::zeros(m_H, m_W, CV_8UC1));
      for (size_t idx = 0; idx < circles.size(); idx++) 
	    cv::drawContours(newgradient, circles, idx, cv::Scalar(255));

      
      //Apply the Hough Transform to find the circles
      readHough("../config/dbh_hough.par");
      vector<cv::Vec3f> houghcircles;
      cv::HoughCircles(newgradient, houghcircles, CV_HOUGH_GRADIENT, m_hoDp, m_hoMinDeRa, m_holowTh, m_hohighTh, m_hoMinRa, m_hoMaxRa);
      
      //drawing and counting number of detected shadows
      cv::Mat resultingbeads = m_inputImage.clone();
      for(size_t i = 0; i < houghcircles.size(); i++)
      {
	  cv::Point center(cvRound(houghcircles[i][0]), cvRound(houghcircles[i][1]));
	  int radius = cvRound(houghcircles[i][2]);
	  circle( resultingbeads, center, radius, cv::Scalar(0,0,255), 1, 2, 0 );
      }
      if(circles.size()) 
      {
	    std::stringstream sst; sst << "#" << houghcircles.size();
	    cv::putText(resultingbeads, sst.str(), cv::Point(100,200), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 5, cv::Scalar::all(255), 10, 8);
      }      
     
      //show images here
      if(DEBUG) cv::namedWindow("Gradient2", CV_WINDOW_KEEPRATIO);
      if(DEBUG) cv::imshow("Gradient2", newgradient);
      
      return resultingbeads;
}
//---------------------------------------------------------------------------------
cv::Mat bammPoC::ShadowCounter::DetectByBGS(cv::Mat backgroundroi, vector<cv::Vec3f>& circles_bgs)
{
      //subtract background from ROI
      cv::Mat beadsmask;
      beadsmask = backgroundroi - m_background;
      
      //here we have some simple improvements
      beadsmask = this->Dilate(this->Erode(beadsmask));
      
      //extract shadows using a countor methods
      std::vector<std::vector<cv::Point> > contours;
      cv::findContours(beadsmask.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

      //polynomial curve approximation with closed vertices
      std::vector<std::vector<cv::Point> > contoursPolyApprox;
      for (size_t idx = 0; idx < contours.size(); idx++)
      {
	    std::vector<cv::Point> polyApprox;
	    cv::approxPolyDP(contours.at(idx), polyApprox, 0.3, true);
	    contoursPolyApprox.push_back(polyApprox);
      }
      
      //detect only circles
      std::vector<std::vector<cv::Point> > circles = this->DetectCircles(contoursPolyApprox);
      
      //draw detected circles      
      cv::Mat resultingbeads = m_inputImage.clone();
      for (size_t idx = 0; idx < circles.size(); idx++) 
	    cv::drawContours(resultingbeads, circles, idx, m_rancolors[idx % 128]);
      if(circles.size()) 
      {
	    std::stringstream sst; sst << "#" << circles.size();
	    cv::putText(resultingbeads, sst.str(), cv::Point(100,200), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 5, cv::Scalar::all(255), 10, 8);
      }      
      
      //get circle center and radius
      circles_bgs = this->GetPolygonCenters(circles);      

      //show images here
      if(DEBUG) cv::namedWindow("Gradient3", CV_WINDOW_KEEPRATIO);  
      if(DEBUG) cv::imshow("Gradient3", beadsmask);

      return resultingbeads;
}
//---------------------------------------------------------------------------------
cv::Mat bammPoC::ShadowCounter::DetectByBGSCanny(cv::Mat backgroundroi, vector<cv::Vec3f>& circles_bgscanny)
{
      //subtract background from ROI
      cv::Mat beadsmask;
      beadsmask = backgroundroi - m_background;
      
      //here we have some simple improvements
      //beadsmask = this->Dilate(this->Erode(beadsmask));
      beadsmask = this->Erode(this->Dilate(this->Erode(beadsmask)));
      
      //detect using method 1
      return this->DetectByGradient(beadsmask, circles_bgscanny);
}
//---------------------------------------------------------------------------------
cv::Mat bammPoC::ShadowCounter::GetBackgroundROIOpenCV(unsigned int y, unsigned int x)
{
      //is there a new position defined by the user ?  
      //so, lets get a new background again or avoid unnecessary processing !      
      if(this->m_newpos)
      {
	  cv::Mat buffer = cv::Mat{m_inputImage.size() + cv::Size{2, 2}, CV_8U, cv::Scalar{0}};
	  cv::Scalar thresArray = cv::Scalar::all(m_backgroundThreshold);
	  cv::floodFill(m_inputImage, buffer, m_pxy, 255, NULL, thresArray, thresArray, 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
	  
	  m_background = buffer({{1, 1}, m_inputImage.size()}).clone();
	  m_newpos=false;
      }
      
      //detect the external contour of the selected shape
      cv::Mat backgroundroi = m_background.clone();
      std::vector<std::vector<cv::Point> > contour;
      cv::findContours(backgroundroi, contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
      cv::drawContours(backgroundroi, contour, -1, cv::Scalar(255), CV_FILLED);
  
      return backgroundroi;    
}
//---------------------------------------------------------------------------------
cv::Mat bammPoC::ShadowCounter::GetBackgroundROI(unsigned int y, unsigned int x)
{
      //is there a new position defined by the user ?  
      //so, lets get a new background again or avoid unnecessary processing !      
      if(this->m_newpos)
      {
	  m_background = this->GetRegionGrowingArea(m_pxy.y, m_pxy.x);
	  m_newpos=false;
      }
      
      //detect the external contour of the selected shape
      cv::Mat backgroundroi = m_background.clone();
      std::vector<std::vector<cv::Point> > contour;
      cv::findContours(backgroundroi, contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
      cv::drawContours(backgroundroi, contour, -1, cv::Scalar(255), CV_FILLED);
  
      return backgroundroi;
}
//---------------------------------------------------------------------------------
cv::Mat bammPoC::ShadowCounter::GetRegionGrowingArea(unsigned int y, unsigned int x)
{
      cv::Mat backgroundImage(cv::Mat::zeros(m_H, m_W, CV_8UC1));
      
      //instance of background subtraction module
      bammPoC::Background background;
      background.setDomainSize(m_W, m_H);
      background.initialize(m_inputImage);
      background.setThreshold(m_backgroundThreshold);
      background.selfAdaptative(true);
      Background::Object currobj;
      background.floodFill(y, x, 1, currobj);
      
      //creating a background image from indexers - white is the ROI
      unsigned char pxfr1 = 255;
      for(size_t i = 0; i<m_H; i++)
      {
	    for(size_t j = 0; j<m_W; j++)
	    {
		  if(background.getIndexMap()[i*m_W+j]!=0)
			backgroundImage.at<unsigned char>(i,j) = pxfr1;
	    }
      }
  
      return backgroundImage;
}
//---------------------------------------------------------------------------------
cv::Mat bammPoC::ShadowCounter::GetGradientCanny(cv::Mat inputImage, const char* filename)
{
      //Gray image
      cv::Mat buff(cv::Mat::zeros(m_inputImage.rows, m_inputImage.cols, CV_8UC1));
      cv::cvtColor(inputImage, buff, CV_BGR2GRAY);

      //Gaussian Blur to reduce noise
      GaussianBlur(buff, buff, cv::Size(3, 3), 2, 2);
	    
      //Canny algorithm
      readCanny(filename);
      cv::Canny(buff, buff, m_cannylowThreshold, m_cannyhighThreshold, m_cannykernel_size, true);

      return buff;
}
//---------------------------------------------------------------------------------
std::vector<std::vector<cv::Point> > bammPoC::ShadowCounter::DetectCircles(std::vector<std::vector<cv::Point> > contours)
{
      this->readCircleParameters("../config/circle.par");
    
      std::vector<std::vector<cv::Point> > circles;
      
      //detect circles     
      for (size_t idx = 0; idx < contours.size(); idx++) 
      {    
	    double area = cv::contourArea(contours.at(idx));
	    CvRect bb = cv::boundingRect(contours.at(idx));
	    double radius = bb.width / 2.0;

	    if (std::fabs(1 - static_cast<double>(bb.width) / bb.height) <= m_circle_proportion &&
		std::fabs(1 - area / (CV_PI * std::pow(radius, 2))) <= m_circle_excentrity && 
		area >= m_circle_area_min && area <= m_circle_area_max)
	      
		circles.push_back(contours.at(idx));
      }
      
      return circles;
}
//---------------------------------------------------------------------------------
vector<cv::Vec3f> bammPoC::ShadowCounter::GetPolygonCenters(std::vector<std::vector<cv::Point> > contours)
{
      vector<cv::Vec3f> centers;
      
      //detect circles     
      for (size_t idx = 0; idx < contours.size(); idx++) 
      {    
	    CvRect bb = cv::boundingRect(contours.at(idx));
	    double radius = (bb.width+bb.height) / 4.0;
	    double xx = bb.x + bb.width / 2.0;
	    double yy = bb.y + bb.height / 2.0;
    
	    centers.push_back(cv::Vec3f(xx,yy,radius));
      }
      
      return centers;
}
//---------------------------------------------------------------------------------
cv::Mat bammPoC::ShadowCounter::Erode(const cv::Mat& image)
{
      cv::Mat output = image.clone();
      cv::erode(output, output, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1)));
      return output;
}
//---------------------------------------------------------------------------------
cv::Mat bammPoC::ShadowCounter::Dilate(const cv::Mat& image)
{
      cv::Mat output = image.clone();
      cv::dilate(output, output, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1)));
      return output;
}
//---------------------------------------------------------------------------------
