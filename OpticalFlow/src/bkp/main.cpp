#include "shadowCounter.hpp"

using namespace std;	
//---------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    //checking system arguments
    if(argc != 2)
    {
        cout <<" Usage: <binary exec> ImagePath" << endl;
        return -1;
    }    
    
    //load input image by argument
    cv::Mat inputImage;
    inputImage = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!inputImage.data){
	std::cout << "\nError, image cannot be loaded..." << std::endl;
	return -1;
    }

    //instance of shadow counter method
    bammPoC::ShadowCounter shadowCounter;
    shadowCounter.SetImage(inputImage);

    //run until... ESC
    bool die=false;
    while (!die) 
    {    
	//execute shadow counter
	shadowCounter.Execute();

	//is ESC pressed ?
	char k = cvWaitKey(1);
	//if(k == 27)
	//if(shadowCounter.GetSeedPoint().x != -1)
	//    break;
    }
    
    return 0;
}
