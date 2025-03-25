#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;	
//---------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    //checking system arguments
    if(argc != 3)
    {
        cout <<" Usage: <binary exec> ImagePath" << endl;
        return -1;
    }    
    
    //load input image by argument
    cv::Mat inputImage;
    cv::VideoCapture cap(argv[1]);

    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    // int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    // double fps = cap.get(cv::CAP_PROP_FPS);

    cv::Point MalditoSeedPoint(10,10);
    int i = 0;
    while(1)
    {
        cv::Mat frame;
        bool ret = cap.read(frame);
        if (!ret) {
            std::cout << "Fim do vídeo ou erro ao ler o frame!" << std::endl;
            break;
        }

        // cv::imshow("Frame", frame );

        cv::Mat buffer = cv::Mat{frame.size() + cv::Size{2, 2}, CV_8U, cv::Scalar{0}};
        cv::Scalar thresArray = cv::Scalar::all(5);
        cv::floodFill(frame, buffer, MalditoSeedPoint, 255, NULL, thresArray, thresArray, 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
        cv::Mat m_background = buffer({{1, 1}, frame.size()}).clone();

        // cv::imshow( "Display window", m_background );
        // cv::waitKey(1);

        // char c=(char) cv::waitKey(1);
        // if(c==27)
        //     break;
        
        string f_out = argv[2];
        cv::imwrite(f_out + to_string(19) + "_" + to_string(i) + ".jpg",m_background);
        i++;
    }

    return 0;
}
