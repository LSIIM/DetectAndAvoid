#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <math.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <ctime>

// #define K 5
#define cluster_interations 5
#define PI 3.14159265
using namespace cv;
using namespace std;

struct centroide
{
    Point2f pos;
    Point2f vec_uv;
    Point2f linhas;
    float n_pontos = 0;
};

double cot(double j, double i) {
    if(i == 0){
        if(j > 0)
            return PI/2.0;
        else if(j < 0)
            return -PI/2.0;
        else
            return 0;
    }else if(i < 0){
        if(j >= 0)
            return cosh(j/i)/sinh(j/i) + PI;
        else
            return cosh(j/i)/sinh(j/i) - PI;
    }else
        return cosh(j/i)/sinh(j/i);
}

//vector<int> clusterizar(vector<Point2f> pts, vector<Point2f> linhas, vector<Point2f> uvs, centroide centroides[K]);

int main(int argc, char **argv)
{
    if(argc < 2)
    return 0;

    srand(time(0));
    int K = stoi(argv[1]);

    static unsigned int number_clusters = K;
    float fuzziness = 2.0;    // initial: 1.1
    float epsilon = 0.01;
    SoftCDistType dist_type = kSoftCDistL2;
    SoftCInitType init_type = kSoftCInitRandom;

    VideoCapture capture;
    if (argc > 2)
        capture = VideoCapture(argv[2]);
    else
        capture = VideoCapture(0);

    if (!capture.isOpened()) {
        cerr << "Unable to open file!" << endl;
        return 0;
    }

    int frame_width = static_cast<int>(capture.get(CAP_PROP_FRAME_WIDTH));
    double frame_height = static_cast<int>(capture.get(CAP_PROP_FRAME_HEIGHT));
    double fps = capture.get(CAP_PROP_FPS) || 1;
    double resize_scale = 650.0 / frame_height;

    vector<Scalar> colors;
    RNG rng;
    for(int i = 0; i < 100; i++) {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }

    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;
    capture >> old_frame;
    resize(old_frame, old_frame, Size(), resize_scale, resize_scale);
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

    cv::cuda::GpuMat d_old_gray;
    d_old_gray.upload(old_gray);

    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

    Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK = cuda::SparsePyrLKOpticalFlow::create();
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
    vector<Point2f> paths;

    for (int iter = 0;; iter++) {
        Mat frame, frame_gray;
        capture >> frame;
        if (frame.empty()) break;

        resize(frame, frame, Size(), resize_scale, resize_scale);
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        cv::cuda::GpuMat d_frame, d_gray;
        d_frame.upload(frame);
        d_gray.upload(frame_gray);

        cuda::GpuMat d_p0(p0), d_p1, d_status, d_err;
        d_pyrLK->calc(d_old_gray, d_gray, d_p0, d_p1, d_status, d_err);
        d_p1.download(p1);

        vector<uchar> status;
        d_status.download(status);

        vector<Point2f> good_new;
        vector<Point2f> uvs;

        for (uint i = 0, j = 0; i < p0.size(); i++, j++) {
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                Point2f uv = (p1[i] - p0[i]) * fps;
                uvs.push_back(uv);
                float agl = cot(p1[i].y, p1[i].x);

                if (paths.size() <= j && iter == 0) {
                    paths.push_back((p1[j] - p0[j]) * fps);
                } else {
                    paths[j] = ((paths[j] * (float)iter) + ((p1[i] - p0[i]) * fps)) / (float)(iter + 1);
                }
            } else if (iter != 0) {
                paths.erase(paths.begin() + j);
                j--;
            }
        }

        Mat dataset(Size(5, good_new.size()), CV_32F);
        for (int i = 0; i < good_new.size(); i++) {
            dataset.at<float>(i,0) = paths[i].x;
            dataset.at<float>(i,1) = paths[i].y;
            dataset.at<float>(i,2) = uvs[i].x;
            dataset.at<float>(i,3) = uvs[i].y;
            dataset.at<float>(i,4) = good_new[i].y;
        }

        SoftC::Fuzzy f(dataset, number_clusters, fuzziness, epsilon, dist_type, init_type);
        f.updateRows(dataset);
        f.clustering(100);

        vector<int> minhas_cores;
        cv::Mat memberships = f.get_membership_();
        cv::Mat centroides = f.get_centroids_();

        for (int i = 0; i < memberships.rows; i++) {
            float high = -1;
            int id = 0;
            for (int j = 0; j < memberships.cols; j++) {
                if (memberships.at<float>(i,j) > high) {
                    high = memberships.at<float>(i,j);
                    id = j;
                }
            }
            minhas_cores.push_back(id);
        }

        const float zero = 0.0;
        Mat cluster_pos = Mat_<float>(K, 3, zero);

        for (int i = 0; i < good_new.size(); i++) {
            cluster_pos.at<float>(minhas_cores[i],0) += good_new[i].x;
            cluster_pos.at<float>(minhas_cores[i],1) += good_new[i].y;
            cluster_pos.at<float>(minhas_cores[i],2)++;
            line(mask, p0[i], p1[i], colors[minhas_cores[i]], 1);
            circle(frame, p1[i], 3, colors[minhas_cores[i]], -1);
        }

        float vel_media = 0;
        bool alerta = false;

        for (int i = 0; i < centroides.rows; i++) {
            if (cluster_pos.at<float>(i,2) != 0) {
                cluster_pos.at<float>(i,0) /= cluster_pos.at<float>(i,2);
                cluster_pos.at<float>(i,1) /= cluster_pos.at<float>(i,2);
                Point2f p = Point2f(cluster_pos.at<float>(i,0), cluster_pos.at<float>(i,1));
                circle(frame, p, 7, colors[i], -1);
                arrowedLine(frame, p, (p + Point2f(centroides.at<float>(i,2), centroides.at<float>(i,3))), Scalar(0), 2);

                float velocidade = sqrtf(centroides.at<float>(i,2)*centroides.at<float>(i,2) + centroides.at<float>(i,3)*centroides.at<float>(i,3));
                vel_media += (velocidade * cluster_pos.at<float>(i,2) / (float)good_new.size());

                putText(frame, to_string(velocidade), (p + Point2f(centroides.at<float>(i,2), centroides.at<float>(i,3)) * 1.25), 1, 1, Scalar(255,255,255), 3);
                putText(frame, to_string(velocidade), (p + Point2f(centroides.at<float>(i,2), centroides.at<float>(i,3)) * 1.25), 1, 1, Scalar(0), 2);
            }
        }

        for (int i = 0; i < centroides.rows; i++) {
            float velocidade = sqrtf(centroides.at<float>(i,2)*centroides.at<float>(i,2) + centroides.at<float>(i,3)*centroides.at<float>(i,3));
            if (velocidade - vel_media >= 60) {
                alerta = true;
                Point p(cluster_pos.at<float>(i,0), cluster_pos.at<float>(i,1));
                arrowedLine(frame, p, p + Point(centroides.at<float>(i,2), centroides.at<float>(i,3)), Scalar(0,0,200), 2);
            }
        }

        if (alerta)
            rectangle(frame, Rect(0, 0, frame.cols, frame.rows), Scalar(0, 0, 200), 7);

        Mat img;
        add(frame, mask, img);
        imshow("Frame", img);

        int keyboard = waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
            break;

        d_old_gray = d_gray.clone();
        p0 = good_new;

        if (iter >= 2 * fps - 1) {
            vector<Point2f> tmp_p;
            goodFeaturesToTrack(old_gray, tmp_p, 100, 0.3, 7, Mat(), 7, false, 0.04);
            for (int i = 0; i < tmp_p.size(); i++) {
                bool add = true;
                for (int j = 0; j < p1.size(); j++) {
                    if (norm(p1[j] - tmp_p[i]) < 4)
                        add = false;
                }
                if (add)
                    p0.push_back(tmp_p[i]);
            }
            iter = -1;
            paths.clear();
            mask = Mat::zeros(old_frame.size(), old_frame.type());
        }
    }

    capture.release();
    //  writer.release();

    return 0;
}

// vector<int> clusterizar(vector<Point2f> pts, vector<Point2f> linhas, vector<Point2f> uvs, centroide centroides[K]){
//     vector<int> grupos;
//     for(int i = 0; i < cluster_interations; i++) {
//         for(int j = 0; j < pts.size(); j++){
//             float min = INFINITY;
//             int id = -1;

//             for(int l = 0; l < K; l++){
//                 float dist_centroid = sqrtf(//powf((pts[j].x - centroides[l].pos.x)/10.0,2)
//                                         //powf((pts[j].y - centroides[l].pos.y)*10,2)
//                                          powf(linhas[j].x - centroides[l].linhas.x,2)
//                                         + powf(linhas[j].y - centroides[l].linhas.y,2)
//                                         + powf(uvs[j].x - centroides[l].vec_uv.x,2)
//                                         + powf(uvs[j].y - centroides[l].vec_uv.y,2));
//                 if(dist_centroid < min){
//                     min = dist_centroid;
//                     id = l;
//                 }
//             }
//             grupos.push_back(id);
//         }
//         for(int c = 0; c < K; c++) centroides[c].n_pontos = 0;

//         for(int j = 0; j < pts.size(); j++){
//             if(centroides[grupos[j]].n_pontos == 0){
//                 centroides[grupos[j]].pos = pts[j];
//                 centroides[grupos[j]].vec_uv = uvs[j];
//                 centroides[grupos[j]].linhas = linhas[j];
//             }
//             else{
//                 centroides[grupos[j]].pos = ((centroides[grupos[j]].pos * centroides[grupos[j]].n_pontos) + pts[j]) / (centroides[grupos[j]].n_pontos + 1.0);
//                 centroides[grupos[j]].vec_uv = ((centroides[grupos[j]].vec_uv * centroides[grupos[j]].n_pontos) + uvs[j]) / (centroides[grupos[j]].n_pontos + 1.0);
//                 centroides[grupos[j]].linhas = ((centroides[grupos[j]].linhas * centroides[grupos[j]].n_pontos) + linhas[j]) / (centroides[grupos[j]].n_pontos + 1.0);
//             }
//             centroides[grupos[j]].n_pontos++;
//         }

//     }   

//     return grupos;
// }

