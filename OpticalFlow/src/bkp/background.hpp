#ifndef _BACKGROUND_H
#define	_BACKGROUND_H

#include <cv.h>
#include <iostream>
#include <vector>

using std::vector;
enum seedmoving {N,E,S,W,NE,SE,SW,NW,NONE};

namespace bammPoC{

class Background
{
public:
    //object used to represent a label
    struct Object
    {
	//object indexes
	unsigned int obindex;      
        //geometric center of object
        cv::Point2f gcenter;        
        //area in pixels of the object
        unsigned int pxarea;        
        //bounding box denoting the object limits [0]-init X and Y, [1]-final X and Y
        cv::Point2f boundbox[2];        
        //mean color of the object
        cv::Vec3b meancolor;      
    };

    //default constructors and destructor
    Background() : m_eightCnx(false), m_adaptivePixel(false), m_domainIndex(NULL), m_th(0) {};
    ~Background();
    
    //set domain size
    void setDomainSize(unsigned int width, unsigned int height) {m_width=width; m_height=height; };

    //initialize when only use floodFill method
    void initialize(cv::Mat bgsframe);
    
    //performs FLOOD fill operation over coordinates i and j
    void floodFill(unsigned int i, unsigned int j, unsigned int indexob, Background::Object& currobj);

    //evaluate seed movents aroind i and j
    seedmoving evaluateSeed(unsigned int i, unsigned int j);
    
    //change seed to i and j, as well as its corresponding boundingbox and meancolor
    void moveSeed(unsigned int& i, unsigned int& j, seedmoving moveto, Background::Object& currobj);
    
    //Execute background
    void execute(cv::Mat bsgframe);

    //return a list of objects within minArea variable
    vector<Background::Object> getObjects(int minArea=0, int maxArea=-1);
    
    //return index map
    unsigned int* getIndexMap() const { return m_domainIndex; };
    
    //should be used eight connections ?
    void eightConnections(bool cnx);
    
    //set thresholding value
    void setThreshold(unsigned int thres = 10) { m_th=thres; };
    
    //should the method work with an adaptative pixel reference ?
    void selfAdaptative(bool self=true) { m_adaptivePixel= self; };    
    
private:
    //eight connections ?
    bool m_eightCnx;

    //navigation mode adaptive or only threshold
    bool m_adaptivePixel;
        
    //planar domain size
    unsigned int m_width, m_height;
    
    //Pointer to bsg frame and original frame
    cv::Mat m_inputImage;
    
    //planar domain of object indexes (0-background,1-object,2-object,3-object...)
    unsigned int* m_domainIndex;
    
    // Threshold variable used to refine bgs filter input
    int m_th;
    
    //reference intensity used to navigate on map
    cv::Vec3b m_colorRef;
     
    //list of extracted objects from input frame    
    vector<Background::Object> m_objectList;   
};

};

#endif
