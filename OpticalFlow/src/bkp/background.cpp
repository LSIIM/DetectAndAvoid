#include "background.hpp"
#include <highgui.h>

//---------------------------------------------------------------------------------
bammPoC::Background::~Background()
{
    if(m_domainIndex) free(m_domainIndex);
    m_objectList.clear();
}
//---------------------------------------------------------------------------------
void bammPoC::Background::eightConnections(bool cnx)
{
    m_eightCnx = cnx;
}
//---------------------------------------------------------------------------------
void bammPoC::Background::initialize(cv::Mat bgsframe)
{
    //was domain size defined ?
    m_inputImage = bgsframe.clone();
    
    m_width =m_inputImage.cols;
    m_height=m_inputImage.rows;
    if(!m_width || !m_height) 
    {
	  std::cout << "Error, image domain is zero in at least one dimension" << std::endl;
	  return;
    }
    
    //was m_domainIndex index created ?
    if(!m_domainIndex) m_domainIndex = (unsigned int*) calloc(m_width*m_height, sizeof(unsigned int));
    
    //clear index map buffer
    register unsigned int* p;
    unsigned int domainsize=m_width*m_height;
    for(p=m_domainIndex; p < m_domainIndex+domainsize; p++)
	  *p = 0;

    //erase object list to receive new ones
    m_objectList.clear();  
}
//---------------------------------------------------------------------------------
void bammPoC::Background::execute(cv::Mat inputImage)
{
    initialize(inputImage);
  
    //do labeling over all image pixels
    unsigned int obindex=1;
    for (size_t i = 0; i < static_cast<size_t>(m_inputImage.rows); ++i)
    {
	for (size_t j = 0; j < static_cast<size_t>(m_inputImage.cols); ++j)
	{
            unsigned int idomain=i*m_width+j;
            if(m_domainIndex[idomain] == 0)
            {
                //create a new object
                Background::Object obj;
		
                //flood fill it
                floodFill(i,j,obindex,obj);

                //pushback it into the list
                m_objectList.push_back(obj);
                obindex++;
            }
        }
    }

    std::cout << "number of objects found: " << m_objectList.size() << std::endl;
}
//---------------------------------------------------------------------------------
void bammPoC::Background::floodFill(unsigned int i, unsigned int j, unsigned int indexob, Background::Object& currobj)
{
    if((i<0 || i>m_inputImage.rows) || (j<0 || j>m_inputImage.cols))
    {
	  printf("\nError, flood is out of image bounds...");
	  return;
    }
  
    //initializing current object
    currobj.boundbox[0].x=currobj.boundbox[1].x=j;
    currobj.boundbox[0].y=currobj.boundbox[1].y=i;
    currobj.gcenter = cv::Point2f(j,i);
    currobj.pxarea = 1;
    currobj.obindex=indexob;
    
    //temporary variables
    unsigned long gmean[2] = {j,i};
    unsigned long gcolor[3] = {0,0,0};
    
    //starting seed point with current index
    unsigned int idomain=i*m_width+j;
    int prevIndex=m_domainIndex[idomain];
    m_domainIndex[idomain]=indexob;
    
    cv::Vec3b pxfr = m_inputImage.at<cv::Vec3b>(i,j);
    m_colorRef=pxfr;
		
    //is this point The Last of the Mohicans ?
    seedmoving moveto = evaluateSeed(i,j);
    if(moveto == NONE)
    {        
	currobj.pxarea = 0;
	m_domainIndex[idomain]=prevIndex;
	currobj.obindex=prevIndex;
        return;
    }

    //store seed in stack path
    vector<cv::Point> recxy;
    recxy.push_back(cv::Point(j,i));

    //change seed position
    moveSeed(i,j,moveto,currobj);
    gmean[0] += j;
    gmean[1] += i;
    currobj.pxarea++;

    //FLOOD AROUND IT (i,j) !!!!
    //stack needed to avoid recursion and recovery previous valid x and y
    while(recxy.size() > 0)
    {
        //planar domain receices current index
        idomain=i*m_width+j;
        m_domainIndex[idomain]=indexob;

	if(m_adaptivePixel)
	{
	      cv::Vec3b pxfr = m_inputImage.at<cv::Vec3b>(i,j);
	      m_colorRef=pxfr;
	}

	//find the next position based on current seed point
        moveto = evaluateSeed(i,j);

        //cv::Point p = recxy.back();
        //if((moveto!=NONE) && (p.x != j || p.y != i))
        if(moveto!=NONE)
        {
            //store seed in stack path
            recxy.push_back(cv::Point(j,i));

            //change seed position
            moveSeed(i,j,moveto,currobj);

            gmean[0] += j;
            gmean[1] += i;
            currobj.pxarea++;
        }
        else
        {
            //get last element valid from stack
            //to try find a valid position from there
            cv::Point p = recxy.back();
            recxy.pop_back();
            j = p.x;
            i = p.y;            
        }        
    }
    
    //area and center of the object
    currobj.gcenter.x = gmean[0] / currobj.pxarea;
    currobj.gcenter.y = gmean[1] / currobj.pxarea;

    //mean color of the object
    currobj.meancolor[0] = gcolor[0] / currobj.pxarea;
    currobj.meancolor[1] = gcolor[1] / currobj.pxarea;
    currobj.meancolor[2] = gcolor[2] / currobj.pxarea;

    //#ifndef NDEBUG
    //std::cout << "Object " << indexob << " with area " << currobj.pxarea << " and center at " << currobj.gcenter.x << "-" << currobj.gcenter.y << std::endl;
    //std::cout << "   BoundingBox starting at " << currobj.boundbox[0].x << "-" << currobj.boundbox[0].y << " to " << currobj.boundbox[1].x << "-" << currobj.boundbox[1].y << std::endl;
    //std::cout << "   Mean color is " << currobj.meancolor[0] << "-" << currobj.meancolor[1] << "-" << currobj.meancolor[2] << std::endl;
    //#endif    
}
//---------------------------------------------------------------------------------
seedmoving bammPoC::Background::evaluateSeed(unsigned int i, unsigned int j)
{
    //coordinating system is
    //N,E,S,W,NE,SE,SW,NW
        
    int di = (int) i;
    int dj = (int) j;                       
    
    //can the seed point be moved to NORTH (N)?
    if(di-1 > 0)
    {
        unsigned int idomain=(i-1)*m_width+j;
        cv::Vec3b pxfr = m_inputImage.at<cv::Vec3b>(i-1,j);
	float de = sqrt((pxfr[0]-m_colorRef[0])*(pxfr[0]-m_colorRef[0])+(pxfr[1]-m_colorRef[1])*(pxfr[1]-m_colorRef[1])+(pxfr[2]-m_colorRef[2])*(pxfr[2]-m_colorRef[2]));
        if(m_domainIndex[idomain] == 0 && de < m_th)
            return N;
    }
    //can the seed point be moved to LEFT (E)?
    if(dj+1 < (int)m_width)
    {
        unsigned int idomain=(i)*m_width+(j+1);
        cv::Vec3b pxfr = m_inputImage.at<cv::Vec3b>(i,j+1);
	float de = sqrt((pxfr[0]-m_colorRef[0])*(pxfr[0]-m_colorRef[0])+(pxfr[1]-m_colorRef[1])*(pxfr[1]-m_colorRef[1])+(pxfr[2]-m_colorRef[2])*(pxfr[2]-m_colorRef[2]));
        if(m_domainIndex[idomain] == 0 && de < m_th)
            return E;
    }
    //can the seed point be moved to DOWN (S)?
    if(di+1 < (int)m_height)
    {
        unsigned int idomain=(i+1)*m_width+j;
        cv::Vec3b pxfr = m_inputImage.at<cv::Vec3b>(i+1,j);
	float de = sqrt((pxfr[0]-m_colorRef[0])*(pxfr[0]-m_colorRef[0])+(pxfr[1]-m_colorRef[1])*(pxfr[1]-m_colorRef[1])+(pxfr[2]-m_colorRef[2])*(pxfr[2]-m_colorRef[2]));
        if(m_domainIndex[idomain] == 0 && de < m_th)
            return S;
    }
    //can the seed point be moved to LEFT (W)?
    if(dj-1 > 0)
    {
        unsigned int idomain=i*m_width+(j-1);
        cv::Vec3b pxfr = m_inputImage.at<cv::Vec3b>(i,j-1);
	float de = sqrt((pxfr[0]-m_colorRef[0])*(pxfr[0]-m_colorRef[0])+(pxfr[1]-m_colorRef[1])*(pxfr[1]-m_colorRef[1])+(pxfr[2]-m_colorRef[2])*(pxfr[2]-m_colorRef[2]));
        if(m_domainIndex[idomain] == 0 && de < m_th)
            return W;
    }

    if(m_eightCnx)
    {
	//can the seed point be moved to (NE)?
	if(di-1 > 0 && dj+1 < (int)m_width)
	{
	    unsigned int idomain=(i-1)*m_width+(j+1);
	    cv::Vec3b pxfr = m_inputImage.at<cv::Vec3b>(i-1,j+1);
	    float de = sqrt((pxfr[0]-m_colorRef[0])*(pxfr[0]-m_colorRef[0])+(pxfr[1]-m_colorRef[1])*(pxfr[1]-m_colorRef[1])+(pxfr[2]-m_colorRef[2])*(pxfr[2]-m_colorRef[2]));
	    if(m_domainIndex[idomain] == 0 && de < m_th)
		return NE;
	}
	//can the seed point be moved to (SE)?
	if(di+1 < (int)m_height && dj+1 < (int)m_width) 
	{
	    unsigned int idomain=(i+1)*m_width+(j+1);
	    cv::Vec3b pxfr = m_inputImage.at<cv::Vec3b>(i+1,j+1);
	    float de = sqrt((pxfr[0]-m_colorRef[0])*(pxfr[0]-m_colorRef[0])+(pxfr[1]-m_colorRef[1])*(pxfr[1]-m_colorRef[1])+(pxfr[2]-m_colorRef[2])*(pxfr[2]-m_colorRef[2]));
	    if(m_domainIndex[idomain] == 0 && de < m_th)
		return SE;
	}
	//can the seed point be moved to (SW)?
	if(di+1 < (int)m_height && dj-1 > 0)
	{
	    unsigned int idomain=(i+1)*m_width+(j-1);
	    cv::Vec3b pxfr = m_inputImage.at<cv::Vec3b>(i+1,j-1);
	    float de = sqrt((pxfr[0]-m_colorRef[0])*(pxfr[0]-m_colorRef[0])+(pxfr[1]-m_colorRef[1])*(pxfr[1]-m_colorRef[1])+(pxfr[2]-m_colorRef[2])*(pxfr[2]-m_colorRef[2]));
	    if(m_domainIndex[idomain] == 0 && de < m_th)
		return SW;
	}
	//can the seed point be moved to (NW)?
	if(di-1 > 0 && dj-1 > 0)
	{
	    unsigned int idomain=(i-1)*m_width+(j-1);
	    cv::Vec3b pxfr = m_inputImage.at<cv::Vec3b>(i-1,j-1);
	    float de = sqrt((pxfr[0]-m_colorRef[0])*(pxfr[0]-m_colorRef[0])+(pxfr[1]-m_colorRef[1])*(pxfr[1]-m_colorRef[1])+(pxfr[2]-m_colorRef[2])*(pxfr[2]-m_colorRef[2]));
	    if(m_domainIndex[idomain] == 0 && de < m_th)
		return NW;
	}        
    }

    //else, no valid position was found, return NONE
    return NONE;
}
//---------------------------------------------------------------------------------
void bammPoC::Background::moveSeed(unsigned int& i, unsigned int& j, seedmoving moveto, Background::Object& currobj)
{
    if(moveto == NONE) return;

    switch((int)moveto)
    {
    case N :
        i--;
        break;
    case E :
        j++;
        break;
    case S :
        i++;
        break;
    case W :
        j--;
        break;
    }

    //is there need to perform 8 adjacent conections
    if(m_eightCnx)
    {
        switch((int)moveto)
        {
        case NE :
            i--;j++;
            break;
        case SE :
            i++;j++;
            break;
        case SW :
            i++;j--;
            break;
        case NW :
            i--;j--;
            break;
        }
    }
    //adjust bounding box
    currobj.boundbox[0].y=(i<currobj.boundbox[0].y) ? i : currobj.boundbox[0].y;
    currobj.boundbox[1].y=(i>currobj.boundbox[1].y) ? i : currobj.boundbox[1].y;
    currobj.boundbox[0].x=(j<currobj.boundbox[0].x) ? j : currobj.boundbox[0].x;
    currobj.boundbox[1].x=(j>currobj.boundbox[1].x) ? j : currobj.boundbox[1].x;
    
    if(m_adaptivePixel)
    {
	  cv::Vec3b pxfr = m_inputImage.at<cv::Vec3b>(i,j);
	  m_colorRef=pxfr;
    }
}
//---------------------------------------------------------------------------------
vector<bammPoC::Background::Object> bammPoC::Background::getObjects(int _minArea, int _maxArea)
{
    unsigned int minArea, maxArea;
    minArea = (_minArea < 0) ? 0  : _minArea;
    maxArea = (_maxArea < 0) ? -1 : _maxArea;
  
    vector<Background::Object> resultList;
    
    for(unsigned int i=0; i< m_objectList.size(); i++)
    {
        if(m_objectList.at(i).pxarea >= minArea && m_objectList.at(i).pxarea <= maxArea)
        {
            resultList.push_back(m_objectList.at(i));
        }        
    }

    return resultList;
}
//---------------------------------------------------------------------------------
