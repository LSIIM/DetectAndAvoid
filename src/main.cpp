#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#if defined(DEA_ENABLE_TENSORRT) && DEA_ENABLE_TENSORRT
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#if defined(DEA_ENABLE_TRT_ONNX_PARSER) && DEA_ENABLE_TRT_ONNX_PARSER
#include <NvOnnxParser.h>
#endif
#endif

#if defined(DEA_ENABLE_VPI) && DEA_ENABLE_VPI
#include <vpi/OpenCVInterop.hpp>
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr const char* kTrackerConfig = "bytetrack.yaml";

fs::path projectRootPath() {
#ifdef DEA_CPP_SOURCE_DIR
    return fs::path(DEA_CPP_SOURCE_DIR);
#else
    return fs::current_path();
#endif
}

std::string defaultYoloModelPath() {
    fs::path p = projectRootPath() / "models" / "best_yolo_11_JUNHO_nano_drones_DGX_rebuilt.engine";
    return fs::weakly_canonical(p).string();
}

std::string defaultSkyModelPath() {
    const fs::path engine_p = projectRootPath() / "models" / "skyseg_fp16_trt_sm87.engine";
    if (fs::exists(engine_p)) {
        return fs::weakly_canonical(engine_p).string();
    }
    const fs::path onnx_p = projectRootPath() / "models" / "skyseg_fp16.onnx";
    return fs::weakly_canonical(onnx_p).string();
}

double nowSeconds() {
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - start);
    return elapsed.count();
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string truncateText(const std::string& text, std::size_t max_len) {
    if (text.size() <= max_len) {
        return text;
    }
    if (max_len < 3) {
        return text.substr(0, max_len);
    }
    return text.substr(0, max_len - 3) + "...";
}

void putShadowText(
    cv::Mat& frame,
    const std::string& text,
    const cv::Point& org,
    double scale = 0.6,
    const cv::Scalar& fg = cv::Scalar(255, 255, 255),
    const cv::Scalar& bg = cv::Scalar(0, 0, 0),
    int thick = 2
) {
    cv::putText(frame, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, bg, thick + 1, cv::LINE_AA);
    cv::putText(frame, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv::LINE_AA);
}

cv::Mat ensure3ch(const cv::Mat& frame, const cv::Mat& fallback) {
    if (frame.empty()) {
        return fallback;
    }
    if (frame.channels() == 1) {
        cv::Mat out;
        cv::cvtColor(frame, out, cv::COLOR_GRAY2BGR);
        return out;
    }
    if (frame.channels() == 4) {
        cv::Mat out;
        cv::cvtColor(frame, out, cv::COLOR_BGRA2BGR);
        return out;
    }
    return frame;
}

float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

struct Args {
    std::string video_file;
    std::string video_ip = "192.168.144.25";
    int video_port = 1945;
    std::string video_path = "/";
    std::string rtsp_backend = "auto";    // auto | gstreamer | ffmpeg
    std::string rtsp_transport = "tcp";   // tcp | udp | auto
    int rtsp_latency_ms = 80;
    int rtsp_open_timeout_ms = 2500;
    double rtsp_first_frame_timeout = 15.0;
    int rtsp_max_consecutive_timeouts = 6;

    int resize_height = 360;
    int clusters = 3;
    float confidence = 0.6F;

    std::string yolo_model_path = defaultYoloModelPath();
    std::string sky_model_path = defaultSkyModelPath();
    bool disable_sky = false;
    bool disable_flow = false;
    bool flow_gpu = false;

    int yolo_update_interval = 2;
    int sky_update_interval = 3;
    int flow_update_interval = 1;

    std::string output;
    double output_fps = 30.0;
    bool no_display = false;
    double stats_interval = 2.0;
    double read_timeout = 2.0;
};

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Input options:\n"
        << "  --video-file <path>          Input video file path\n"
        << "  --video-ip <ip>              RTSP stream IP (default: 192.168.144.25)\n"
        << "  --video-port <port>          RTSP stream port (default: 1945)\n"
        << "  --video-path <path>          RTSP path (default: /)\n"
        << "  --rtsp-backend <mode>        RTSP backend: auto|gstreamer|ffmpeg (default: auto)\n"
        << "  --rtsp-transport <mode>      RTSP transport: tcp|udp|auto (default: tcp)\n"
        << "  --rtsp-latency-ms <int>      RTSP latency for GStreamer rtspsrc (default: 80)\n"
        << "  --rtsp-open-timeout-ms <int> RTSP open timeout in ms (default: 2500)\n"
        << "  --rtsp-first-frame-timeout <float> Seconds to wait first RTSP frame (default: 15)\n"
        << "  --rtsp-max-timeouts <int>    Consecutive RTSP timeouts before stop (default: 6)\n\n"
        << "Processing options:\n"
        << "  --resize-height <int>        Processing frame height (default: 360)\n"
        << "  --clusters <int>             Optical flow clusters (default: 3)\n"
        << "  --confidence <float>         YOLO confidence threshold (default: 0.6)\n"
        << "  --yolo-model-path <path>     YOLO model (.engine or .onnx)\n"
        << "  --sky-model-path <path>      SkySeg model (.engine or .onnx)\n"
        << "  --disable-sky                Disable sky segmentation\n"
        << "  --disable-flow               Disable optical flow\n"
        << "  --flow-gpu                   Request GPU optical flow (CPU fallback supported)\n\n"
        << "Intervals:\n"
        << "  --yolo-update-interval <int> Run YOLO every N frames\n"
        << "  --sky-update-interval <int>  Run SkySeg every N frames\n"
        << "  --flow-update-interval <int> Run OpticalFlow every N frames\n\n"
        << "Output options:\n"
        << "  --output <path>              Output video path\n"
        << "  --output-fps <float>         Output FPS (default: 30)\n"
        << "  --no-display                 Disable display window\n"
        << "  --stats-interval <float>     Seconds between stats prints\n"
        << "  --read-timeout <float>       Seconds to wait for a new frame\n"
        << "  --help                       Show this help\n";
}

bool parseInt(const std::string& s, int& out) {
    try {
        std::size_t pos = 0;
        const int v = std::stoi(s, &pos);
        if (pos != s.size()) {
            return false;
        }
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseFloat(const std::string& s, float& out) {
    try {
        std::size_t pos = 0;
        const float v = std::stof(s, &pos);
        if (pos != s.size()) {
            return false;
        }
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseDouble(const std::string& s, double& out) {
    try {
        std::size_t pos = 0;
        const double v = std::stod(s, &pos);
        if (pos != s.size()) {
            return false;
        }
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseArgs(int argc, char** argv, Args& args, bool& show_help) {
    show_help = false;

    auto needValue = [&](int idx, const std::string& opt) -> std::optional<std::string> {
        if (idx + 1 >= argc) {
            std::cerr << "Missing value for option: " << opt << '\n';
            return std::nullopt;
        }
        return std::string(argv[idx + 1]);
    };

    for (int i = 1; i < argc; ++i) {
        const std::string opt(argv[i]);

        if (opt == "--help" || opt == "-h") {
            show_help = true;
            return true;
        }
        if (opt == "--disable-sky") {
            args.disable_sky = true;
            continue;
        }
        if (opt == "--disable-flow") {
            args.disable_flow = true;
            continue;
        }
        if (opt == "--flow-gpu") {
            args.flow_gpu = true;
            continue;
        }
        if (opt == "--no-display") {
            args.no_display = true;
            continue;
        }

        auto v = needValue(i, opt);
        if (!v.has_value()) {
            return false;
        }

        if (opt == "--video-file") {
            args.video_file = *v;
            ++i;
            continue;
        }
        if (opt == "--video-ip") {
            args.video_ip = *v;
            ++i;
            continue;
        }
        if (opt == "--video-port") {
            int parsed = 0;
            if (!parseInt(*v, parsed)) {
                std::cerr << "Invalid integer for --video-port: " << *v << '\n';
                return false;
            }
            args.video_port = parsed;
            ++i;
            continue;
        }
        if (opt == "--video-path") {
            args.video_path = *v;
            ++i;
            continue;
        }
        if (opt == "--rtsp-backend") {
            const std::string mode = toLower(*v);
            if (mode != "auto" && mode != "gstreamer" && mode != "ffmpeg") {
                std::cerr << "Invalid value for --rtsp-backend: " << *v << " (use auto|gstreamer|ffmpeg)\n";
                return false;
            }
            args.rtsp_backend = mode;
            ++i;
            continue;
        }
        if (opt == "--rtsp-transport") {
            const std::string mode = toLower(*v);
            if (mode != "tcp" && mode != "udp" && mode != "auto") {
                std::cerr << "Invalid value for --rtsp-transport: " << *v << " (use tcp|udp|auto)\n";
                return false;
            }
            args.rtsp_transport = mode;
            ++i;
            continue;
        }
        if (opt == "--rtsp-latency-ms") {
            int parsed = 0;
            if (!parseInt(*v, parsed) || parsed < 0) {
                std::cerr << "Invalid integer for --rtsp-latency-ms: " << *v << '\n';
                return false;
            }
            args.rtsp_latency_ms = parsed;
            ++i;
            continue;
        }
        if (opt == "--rtsp-open-timeout-ms") {
            int parsed = 0;
            if (!parseInt(*v, parsed) || parsed <= 0) {
                std::cerr << "Invalid integer for --rtsp-open-timeout-ms: " << *v << '\n';
                return false;
            }
            args.rtsp_open_timeout_ms = parsed;
            ++i;
            continue;
        }
        if (opt == "--rtsp-first-frame-timeout") {
            double parsed = 0.0;
            if (!parseDouble(*v, parsed) || parsed <= 0.0) {
                std::cerr << "Invalid float for --rtsp-first-frame-timeout: " << *v << '\n';
                return false;
            }
            args.rtsp_first_frame_timeout = parsed;
            ++i;
            continue;
        }
        if (opt == "--rtsp-max-timeouts") {
            int parsed = 0;
            if (!parseInt(*v, parsed) || parsed <= 0) {
                std::cerr << "Invalid integer for --rtsp-max-timeouts: " << *v << '\n';
                return false;
            }
            args.rtsp_max_consecutive_timeouts = parsed;
            ++i;
            continue;
        }
        if (opt == "--resize-height") {
            int parsed = 0;
            if (!parseInt(*v, parsed)) {
                std::cerr << "Invalid integer for --resize-height: " << *v << '\n';
                return false;
            }
            args.resize_height = parsed;
            ++i;
            continue;
        }
        if (opt == "--clusters") {
            int parsed = 0;
            if (!parseInt(*v, parsed)) {
                std::cerr << "Invalid integer for --clusters: " << *v << '\n';
                return false;
            }
            args.clusters = parsed;
            ++i;
            continue;
        }
        if (opt == "--confidence") {
            float parsed = 0.0F;
            if (!parseFloat(*v, parsed)) {
                std::cerr << "Invalid float for --confidence: " << *v << '\n';
                return false;
            }
            args.confidence = parsed;
            ++i;
            continue;
        }
        if (opt == "--yolo-model-path") {
            args.yolo_model_path = *v;
            ++i;
            continue;
        }
        if (opt == "--sky-model-path") {
            args.sky_model_path = *v;
            ++i;
            continue;
        }
        if (opt == "--yolo-update-interval") {
            int parsed = 0;
            if (!parseInt(*v, parsed)) {
                std::cerr << "Invalid integer for --yolo-update-interval: " << *v << '\n';
                return false;
            }
            args.yolo_update_interval = parsed;
            ++i;
            continue;
        }
        if (opt == "--sky-update-interval") {
            int parsed = 0;
            if (!parseInt(*v, parsed)) {
                std::cerr << "Invalid integer for --sky-update-interval: " << *v << '\n';
                return false;
            }
            args.sky_update_interval = parsed;
            ++i;
            continue;
        }
        if (opt == "--flow-update-interval") {
            int parsed = 0;
            if (!parseInt(*v, parsed)) {
                std::cerr << "Invalid integer for --flow-update-interval: " << *v << '\n';
                return false;
            }
            args.flow_update_interval = parsed;
            ++i;
            continue;
        }
        if (opt == "--output") {
            args.output = *v;
            ++i;
            continue;
        }
        if (opt == "--output-fps") {
            double parsed = 0.0;
            if (!parseDouble(*v, parsed)) {
                std::cerr << "Invalid float for --output-fps: " << *v << '\n';
                return false;
            }
            args.output_fps = parsed;
            ++i;
            continue;
        }
        if (opt == "--stats-interval") {
            double parsed = 0.0;
            if (!parseDouble(*v, parsed)) {
                std::cerr << "Invalid float for --stats-interval: " << *v << '\n';
                return false;
            }
            args.stats_interval = parsed;
            ++i;
            continue;
        }
        if (opt == "--read-timeout") {
            double parsed = 0.0;
            if (!parseDouble(*v, parsed)) {
                std::cerr << "Invalid float for --read-timeout: " << *v << '\n';
                return false;
            }
            args.read_timeout = parsed;
            ++i;
            continue;
        }

        std::cerr << "Unknown option: " << opt << '\n';
        return false;
    }

    return true;
}

void configureRuntime() {
    cv::setUseOptimized(true);
    try {
        const unsigned int cpu = std::thread::hardware_concurrency();
        const int nthreads = static_cast<int>(std::max(1u, std::min(cpu == 0 ? 1u : cpu, 8u)));
        cv::setNumThreads(nthreads);
    } catch (...) {
    }
    try {
        cv::ocl::setUseOpenCL(true);
    } catch (...) {
    }
}

std::string buildRtspUrl(const std::string& ip, int port, const std::string& path) {
    std::string normalized = path;
    if (normalized.empty() || normalized.front() != '/') {
        normalized = "/" + normalized;
    }
    return "rtsp://" + ip + ":" + std::to_string(port) + normalized;
}

std::string buildGstreamerRtspPipeline(
    const std::string& url,
    const std::string& transport,
    int latency_ms,
    int open_timeout_ms,
    const std::string& variant
) {
    std::ostringstream ss;
    ss << "rtspsrc location=\"" << url << "\" latency=" << std::max(0, latency_ms)
       << " drop-on-latency=false";
    if (open_timeout_ms > 0) {
        const int64_t timeout_us = static_cast<int64_t>(open_timeout_ms) * 1000LL;
        ss << " timeout=" << timeout_us << " tcp-timeout=" << timeout_us;
    }
    if (transport == "tcp" || transport == "udp") {
        ss << " protocols=" << transport;
    }

    if (variant == "h264") {
        ss << " ! application/x-rtp,media=video,encoding-name=H264"
           << " ! queue max-size-buffers=8 max-size-bytes=0 max-size-time=0"
           << " ! rtph264depay"
           << " ! h264parse"
           << " ! nvv4l2decoder enable-max-performance=1"
           << " ! nvvidconv ! video/x-raw,format=BGRx"
           << " ! videoconvert ! video/x-raw,format=BGR";
    } else if (variant == "h265") {
        ss << " ! application/x-rtp,media=video,encoding-name=H265"
           << " ! queue max-size-buffers=8 max-size-bytes=0 max-size-time=0"
           << " ! rtph265depay"
           << " ! h265parse"
           << " ! nvv4l2decoder enable-max-performance=1"
           << " ! nvvidconv ! video/x-raw,format=BGRx"
           << " ! videoconvert ! video/x-raw,format=BGR";
    } else {
        ss << " ! application/x-rtp,media=video"
           << " ! queue max-size-buffers=8 max-size-bytes=0 max-size-time=0"
           << " ! decodebin"
           << " ! videoconvert ! video/x-raw,format=BGR";
    }

    ss << " ! appsink sync=false drop=true max-buffers=1";
    return ss.str();
}

void probeCaptureProperties(cv::VideoCapture& cap, double& fps, int& width, int& height, int tries = 40) {
    fps = cap.get(cv::CAP_PROP_FPS);
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (width > 0 && height > 0) {
        return;
    }

    for (int i = 0; i < tries; ++i) {
        cv::Mat frame;
        if (cap.read(frame) && !frame.empty()) {
            width = frame.cols;
            height = frame.rows;
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool openCapture(
    const std::string& url,
    const std::string& rtsp_backend,
    const std::string& rtsp_transport,
    int rtsp_latency_ms,
    int rtsp_open_timeout_ms,
    cv::VideoCapture& cap,
    std::string& backend_used,
    double& fps,
    int& src_w,
    int& src_h,
    std::string& error_msg
) {
    std::vector<std::string> urls{url};
    // Root path (rtsp://ip:port/) is canonical; avoid retrying the same endpoint without slash.
    const std::size_t scheme_pos = url.find("://");
    const std::size_t path_pos = (scheme_pos == std::string::npos) ? url.find('/') : url.find('/', scheme_pos + 3);
    const bool has_non_root_path = path_pos != std::string::npos && (path_pos + 1) < url.size();
    if (!url.empty() && url.back() == '/' && has_non_root_path) {
        urls.push_back(url.substr(0, url.size() - 1));
    }
    std::sort(urls.begin(), urls.end());
    urls.erase(std::unique(urls.begin(), urls.end()), urls.end());

    std::vector<std::string> errors;

    const bool allow_gst = (rtsp_backend != "ffmpeg");
    // Keep FFmpeg fallback even when gstreamer is forced, so capture doesn't die on missing plugins.
    const bool allow_ffmpeg = true;

    std::vector<std::string> transports;
    if (rtsp_transport == "auto") {
        transports = {"tcp", "udp"};
    } else {
        transports = {rtsp_transport};
    }

    auto tryCandidate = [&](cv::VideoCapture&& candidate, const std::string& label, bool skip_probe_read) -> bool {
        if (label.rfind("GStreamer/", 0) != 0) {
            try {
                candidate.set(cv::CAP_PROP_BUFFERSIZE, 1);
            } catch (...) {
            }
        }

        if (!candidate.isOpened()) {
            errors.emplace_back(label + ": open failed");
            return false;
        }

        int w = static_cast<int>(candidate.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(candidate.get(cv::CAP_PROP_FRAME_HEIGHT));
        double local_fps = candidate.get(cv::CAP_PROP_FPS);

        if (!skip_probe_read && (w <= 0 || h <= 0)) {
            probeCaptureProperties(candidate, local_fps, w, h);
        }

        if (w > 0 && h > 0) {
            cap = std::move(candidate);
            backend_used = label;
            fps = local_fps;
            src_w = w;
            src_h = h;
            return true;
        }

        if (skip_probe_read) {
            // Accept GStreamer capture even without immediate dimensions;
            // final dimensions are probed in main loop setup.
            cap = std::move(candidate);
            backend_used = label;
            fps = local_fps;
            src_w = w;
            src_h = h;
            return true;
        }

        errors.emplace_back(label + ": opened but no valid frame size");
        candidate.release();
        return false;
    };

    auto tryGstreamer = [&](const std::string& u) -> bool {
        if (!allow_gst) {
            return false;
        }
        const std::vector<std::string> gst_variants = (rtsp_backend == "gstreamer")
            ? std::vector<std::string>{"h264", "h265", "decodebin"}
            : std::vector<std::string>{"h264"};
        for (const auto& tr : transports) {
            for (const auto& variant : gst_variants) {
                const std::string pipeline = buildGstreamerRtspPipeline(u, tr, rtsp_latency_ms, rtsp_open_timeout_ms, variant);
                std::string label = std::string("GStreamer/") + variant + "/" + tr;
                cv::VideoCapture candidate(pipeline, cv::CAP_GSTREAMER);
                if (tryCandidate(std::move(candidate), label, true)) {
                    return true;
                }
            }
        }
        return false;
    };

    auto tryFfmpeg = [&](const std::string& u) -> bool {
        if (!allow_ffmpeg) {
            return false;
        }
        for (const auto& tr : transports) {
            const int timeout_us = std::max(100, rtsp_open_timeout_ms) * 1000;
            const int max_delay_us = std::max(50, rtsp_open_timeout_ms / 2) * 1000;
            const std::string ffmpeg_opts =
                std::string("rtsp_transport;") + tr +
                "|stimeout;" + std::to_string(timeout_us) +
                "|max_delay;" + std::to_string(max_delay_us) +
                "|fflags;nobuffer|flags;low_delay|reorder_queue_size;0";
            setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", ffmpeg_opts.c_str(), 1);

            cv::VideoCapture candidate(u, cv::CAP_FFMPEG);
            std::string label = "FFmpeg/" + tr;
            if (rtsp_backend == "gstreamer") {
                label = "FFmpeg/" + tr + " (fallback)";
            }
            if (tryCandidate(std::move(candidate), label, false)) {
                return true;
            }
        }
        return false;
    };

    for (const auto& u : urls) {
        if (rtsp_backend == "auto") {
            if (tryFfmpeg(u)) {
                return true;
            }
            if (tryGstreamer(u)) {
                return true;
            }
        } else {
            if (tryGstreamer(u)) {
                return true;
            }
            if (tryFfmpeg(u)) {
                return true;
            }
        }
    }

    std::ostringstream oss;
    if (errors.empty()) {
        oss << "no attempt details";
    } else {
        const std::size_t start = errors.size() > 4 ? errors.size() - 4 : 0;
        for (std::size_t i = start; i < errors.size(); ++i) {
            if (i > start) {
                oss << " | ";
            }
            oss << errors[i];
        }
    }
    error_msg = oss.str();
    return false;
}

bool openFileCapture(
    const std::string& video_file,
    cv::VideoCapture& cap,
    std::string& backend_used,
    double& fps,
    int& src_w,
    int& src_h,
    std::string& error_msg
) {
    const fs::path p = fs::absolute(fs::path(video_file));
    if (!fs::exists(p)) {
        error_msg = "Video file not found: " + p.string();
        return false;
    }

    cv::VideoCapture candidate(p.string());
    if (!candidate.isOpened()) {
        error_msg = "Could not open video file: " + p.string();
        return false;
    }

    int w = 0;
    int h = 0;
    double local_fps = 0.0;
    probeCaptureProperties(candidate, local_fps, w, h);
    if (w <= 0 || h <= 0) {
        error_msg = "Video opened but no valid frame size: " + p.string();
        candidate.release();
        return false;
    }

    cap = std::move(candidate);
    backend_used = std::string("FILE/") + p.filename().string();
    fps = local_fps;
    src_w = w;
    src_h = h;
    return true;
}

double resolveFps(double raw_fps, double fallback = 30.0) {
    if (raw_fps >= 1.0 && raw_fps <= 120.0) {
        return raw_fps;
    }
    return fallback;
}

class LatestFrameReader {
public:
    explicit LatestFrameReader(cv::VideoCapture* cap) : cap_(cap) {}

    void start() {
        stop_.store(false);
        thread_ = std::thread(&LatestFrameReader::run, this);
    }

    void stop() {
        stop_.store(true);
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    bool getLatest(int64_t last_id, double timeout_sec, cv::Mat& frame, int64_t& frame_id, double& ts) {
        auto timeout = std::chrono::duration<double>(timeout_sec);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(timeout);

        std::unique_lock<std::mutex> lock(mutex_);
        while (!stop_.load()) {
            if (!latest_frame_.empty() && latest_id_ != last_id) {
                frame = latest_frame_.clone();
                frame_id = latest_id_;
                ts = latest_ts_;
                return true;
            }
            if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                return false;
            }
        }
        return false;
    }

    int64_t totalRead() const {
        return total_read_.load();
    }

private:
    void run() {
        while (!stop_.load()) {
            cv::Mat frame;
            bool ok = false;
            if (cap_ != nullptr) {
                ok = cap_->read(frame);
            }
            if (!ok || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            std::lock_guard<std::mutex> lock(mutex_);
            latest_frame_ = frame;
            latest_id_ += 1;
            latest_ts_ = nowSeconds();
            total_read_.fetch_add(1);
            cv_.notify_all();
        }
    }

    cv::VideoCapture* cap_{nullptr};
    std::atomic<bool> stop_{false};
    std::thread thread_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;

    cv::Mat latest_frame_;
    int64_t latest_id_{0};
    double latest_ts_{0.0};
    std::atomic<int64_t> total_read_{0};
};

struct WorkerSnapshot {
    cv::Mat frame;
    int64_t frame_id{0};
    double proc_ms{0.0};
    std::string error;
    int64_t total_processed{0};
};

struct WorkerMetrics {
    int64_t frame_id{0};
    double proc_ms{0.0};
    std::string error;
    int64_t total_processed{0};
};

class ModuleWorker {
public:
    using ProcessFn = std::function<cv::Mat(const cv::Mat&)>;

    ModuleWorker(std::string name, ProcessFn fn)
        : name_(std::move(name)), process_fn_(std::move(fn)) {}

    ~ModuleWorker() {
        stop();
    }

    void start() {
        stop_.store(false);
        thread_ = std::thread(&ModuleWorker::run, this);
    }

    void stop() {
        stop_.store(true);
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void submit(const cv::Mat& frame, int64_t frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_frame_ = frame.clone();
        pending_frame_id_ = frame_id;
        has_pending_ = true;
        cv_.notify_all();
    }

    WorkerSnapshot getLatestOutput() const {
        std::lock_guard<std::mutex> lock(mutex_);
        WorkerSnapshot s;
        s.frame = last_output_.clone();
        s.frame_id = last_output_frame_id_;
        s.proc_ms = last_proc_ms_;
        s.error = last_error_;
        s.total_processed = total_processed_;
        return s;
    }

    WorkerMetrics getLatestMetrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        WorkerMetrics m;
        m.frame_id = last_output_frame_id_;
        m.proc_ms = last_proc_ms_;
        m.error = last_error_;
        m.total_processed = total_processed_;
        return m;
    }

private:
    void run() {
        while (!stop_.load()) {
            cv::Mat frame;
            int64_t frame_id = 0;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(50), [&] { return stop_.load() || has_pending_; });
                if (stop_.load()) {
                    break;
                }
                if (!has_pending_) {
                    continue;
                }
                frame = std::move(pending_frame_);
                frame_id = pending_frame_id_;
                has_pending_ = false;
            }

            const double t0 = nowSeconds();
            cv::Mat out;
            std::string err;
            try {
                out = process_fn_(frame);
            } catch (const std::exception& e) {
                err = e.what();
            } catch (...) {
                err = "unknown exception";
            }
            const double proc_ms = (nowSeconds() - t0) * 1000.0;

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!out.empty()) {
                    last_output_ = out;
                    last_output_frame_id_ = frame_id;
                    total_processed_ += 1;
                }
                last_proc_ms_ = proc_ms;
                last_error_ = err;
            }
        }
    }

    std::string name_;
    ProcessFn process_fn_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::thread thread_;

    cv::Mat pending_frame_;
    int64_t pending_frame_id_{0};
    bool has_pending_{false};

    cv::Mat last_output_;
    int64_t last_output_frame_id_{0};
    double last_proc_ms_{0.0};
    std::string last_error_;
    int64_t total_processed_{0};
};

struct Detection {
    cv::Rect box;
    float confidence{0.0F};
    int class_id{0};
};

struct TrackedDetection {
    Detection det;
    int track_id{0};
    cv::Scalar color;
    std::deque<cv::Point> trail;
};

class SimpleTracker {
public:
    explicit SimpleTracker(int trail_length = 50)
        : trail_length_(trail_length) {}

    std::vector<TrackedDetection> update(const std::vector<Detection>& detections) {
        std::set<int> used_tracks;
        std::set<int> touched_tracks;
        std::vector<TrackedDetection> out;

        for (const auto& det : detections) {
            const cv::Point center(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);

            int best_track = -1;
            double best_dist = max_match_distance_;
            for (auto& [id, tr] : tracks_) {
                if (used_tracks.find(id) != used_tracks.end()) {
                    continue;
                }
                if (tr.miss_count > max_miss_) {
                    continue;
                }
                const double d = cv::norm(center - tr.center);
                if (d < best_dist) {
                    best_dist = d;
                    best_track = id;
                }
            }

            if (best_track < 0) {
                best_track = next_id_++;
                Track t;
                t.id = best_track;
                t.center = center;
                t.box = det.box;
                t.conf = det.confidence;
                t.miss_count = 0;
                t.color = randomColor();
                t.trail.clear();
                t.trail.push_back(center);
                tracks_[best_track] = t;
            } else {
                auto& t = tracks_[best_track];
                t.center = center;
                t.box = det.box;
                t.conf = det.confidence;
                t.miss_count = 0;
                t.trail.push_back(center);
                while (static_cast<int>(t.trail.size()) > trail_length_) {
                    t.trail.pop_front();
                }
            }

            used_tracks.insert(best_track);
            touched_tracks.insert(best_track);

            const auto& t = tracks_.at(best_track);
            TrackedDetection td;
            td.det = det;
            td.track_id = best_track;
            td.color = t.color;
            td.trail = t.trail;
            out.push_back(std::move(td));
        }

        std::vector<int> to_erase;
        for (auto& [id, tr] : tracks_) {
            if (touched_tracks.find(id) == touched_tracks.end()) {
                tr.miss_count += 1;
            }
            if (tr.miss_count > max_miss_) {
                to_erase.push_back(id);
            }
        }
        for (int id : to_erase) {
            tracks_.erase(id);
        }

        return out;
    }

private:
    struct Track {
        int id{0};
        cv::Point center;
        cv::Rect box;
        float conf{0.0F};
        int miss_count{0};
        cv::Scalar color;
        std::deque<cv::Point> trail;
    };

    cv::Scalar randomColor() {
        std::uniform_int_distribution<int> dist(50, 255);
        return cv::Scalar(dist(rng_), dist(rng_), dist(rng_));
    }

    int trail_length_{50};
    int max_miss_{10};
    double max_match_distance_{85.0};
    int next_id_{1};
    std::unordered_map<int, Track> tracks_;
    std::mt19937 rng_{42};
};

#if defined(DEA_ENABLE_TENSORRT) && DEA_ENABLE_TENSORRT

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << '\n';
        }
    }
};

size_t dataTypeSize(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 0;
    }
}

size_t volumeFromDims(const nvinfer1::Dims& dims) {
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return 0;
        }
        v *= static_cast<size_t>(dims.d[i]);
    }
    return v;
}

void checkCuda(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + where + ": " + cudaGetErrorString(err));
    }
}

struct TensorRTOutput {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};

class TensorRTEngine {
public:
    explicit TensorRTEngine(const std::string& engine_path)
        : engine_path_(engine_path) {
        if (!fs::exists(engine_path_)) {
            throw std::runtime_error("YOLO engine not found: " + engine_path_);
        }

        initLibNvInferPlugins(&logger_, "");

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        std::ifstream file(engine_path_, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open engine file: " + engine_path_);
        }
        file.seekg(0, std::ios::end);
        const std::streamsize sz = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> bytes(static_cast<size_t>(sz));
        if (!file.read(bytes.data(), sz)) {
            throw std::runtime_error("Failed to read engine file: " + engine_path_);
        }

        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine_) {
            throw std::runtime_error(
                "TensorRT could not deserialize engine. It may be incompatible with current GPU/CUDA/TensorRT."
            );
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create TensorRT execution context.");
        }

        checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

        const int nb = engine_->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (name == nullptr) {
                continue;
            }
            const auto mode = engine_->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT && input_name_.empty()) {
                input_name_ = name;
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                output_names_.emplace_back(name);
            }
        }

        if (input_name_.empty()) {
            throw std::runtime_error("TensorRT engine has no input tensor.");
        }
        if (output_names_.empty()) {
            throw std::runtime_error("TensorRT engine has no output tensor.");
        }
    }

    ~TensorRTEngine() {
        for (auto& [name, b] : buffers_) {
            (void)name;
            if (b.device_ptr != nullptr) {
                cudaFree(b.device_ptr);
                b.device_ptr = nullptr;
            }
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    static std::string buildEngineFromOnnx(
        const std::string& onnx_path,
        const std::string& engine_out_path,
        int input_w,
        int input_h
    ) {
#if defined(DEA_ENABLE_TRT_ONNX_PARSER) && DEA_ENABLE_TRT_ONNX_PARSER
        if (!fs::exists(onnx_path)) {
            throw std::runtime_error("ONNX file not found for TensorRT build: " + onnx_path);
        }

        TrtLogger logger;
        initLibNvInferPlugins(&logger, "");

        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        if (!builder) {
            throw std::runtime_error("TensorRT createInferBuilder failed.");
        }

        const uint32_t flags = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
        );
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
        if (!network) {
            throw std::runtime_error("TensorRT createNetworkV2 failed.");
        }

        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        if (!parser) {
            throw std::runtime_error("TensorRT ONNX parser creation failed.");
        }

        if (!parser->parseFromFile(
                onnx_path.c_str(),
                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)
            )) {
            std::ostringstream oss;
            oss << "TensorRT ONNX parse failed.";
            const int nerr = parser->getNbErrors();
            for (int i = 0; i < nerr; ++i) {
                const auto* err = parser->getError(i);
                if (err != nullptr) {
                    oss << " [" << i << "] " << err->desc();
                }
            }
            throw std::runtime_error(oss.str());
        }

        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config) {
            throw std::runtime_error("TensorRT createBuilderConfig failed.");
        }

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setBuilderOptimizationLevel(5);
        config->setAvgTimingIterations(4);

        bool has_dynamic = false;
        for (int i = 0; i < network->getNbInputs(); ++i) {
            const auto* in_tensor = network->getInput(i);
            if (in_tensor == nullptr) {
                continue;
            }
            const auto dims = in_tensor->getDimensions();
            for (int d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] < 0) {
                    has_dynamic = true;
                    break;
                }
            }
            if (has_dynamic) {
                break;
            }
        }

        if (has_dynamic) {
            nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
            if (!profile) {
                throw std::runtime_error("TensorRT createOptimizationProfile failed.");
            }

            for (int i = 0; i < network->getNbInputs(); ++i) {
                const auto* in_tensor = network->getInput(i);
                if (in_tensor == nullptr) {
                    continue;
                }

                nvinfer1::Dims min_dims = in_tensor->getDimensions();
                nvinfer1::Dims opt_dims = in_tensor->getDimensions();
                nvinfer1::Dims max_dims = in_tensor->getDimensions();

                for (int d = 0; d < min_dims.nbDims; ++d) {
                    if (min_dims.d[d] >= 0) {
                        continue;
                    }
                    int fallback = 1;
                    if (d == 0) {
                        fallback = 1;
                    } else if (d == 1) {
                        fallback = 3;
                    } else if (d == 2) {
                        fallback = input_h;
                    } else if (d == 3) {
                        fallback = input_w;
                    }
                    min_dims.d[d] = fallback;
                    opt_dims.d[d] = fallback;
                    max_dims.d[d] = fallback;
                }

                if (!profile->setDimensions(
                        in_tensor->getName(),
                        nvinfer1::OptProfileSelector::kMIN,
                        min_dims
                    ) ||
                    !profile->setDimensions(
                        in_tensor->getName(),
                        nvinfer1::OptProfileSelector::kOPT,
                        opt_dims
                    ) ||
                    !profile->setDimensions(
                        in_tensor->getName(),
                        nvinfer1::OptProfileSelector::kMAX,
                        max_dims
                    )) {
                    throw std::runtime_error(
                        std::string("TensorRT setDimensions failed for input: ") + in_tensor->getName()
                    );
                }

                if (in_tensor->isShapeTensor()) {
                    int32_t nvals = 1;
                    const auto sdims = in_tensor->getDimensions();
                    for (int d = 0; d < sdims.nbDims; ++d) {
                        const int dim = sdims.d[d] < 0 ? 1 : sdims.d[d];
                        nvals *= std::max(1, dim);
                    }
                    nvals = std::max(1, nvals);

                    std::vector<int32_t> min_vals(static_cast<size_t>(nvals), 1);
                    std::vector<int32_t> opt_vals(static_cast<size_t>(nvals), 1);
                    std::vector<int32_t> max_vals(static_cast<size_t>(nvals), 1);

                    if (!profile->setShapeValues(
                            in_tensor->getName(),
                            nvinfer1::OptProfileSelector::kMIN,
                            min_vals.data(),
                            nvals
                        ) ||
                        !profile->setShapeValues(
                            in_tensor->getName(),
                            nvinfer1::OptProfileSelector::kOPT,
                            opt_vals.data(),
                            nvals
                        ) ||
                        !profile->setShapeValues(
                            in_tensor->getName(),
                            nvinfer1::OptProfileSelector::kMAX,
                            max_vals.data(),
                            nvals
                        )) {
                        throw std::runtime_error(
                            std::string("TensorRT setShapeValues failed for input: ") + in_tensor->getName()
                        );
                    }
                }
            }

            if (!profile->isValid()) {
                throw std::runtime_error("TensorRT optimization profile is invalid.");
            }

            const int profile_index = config->addOptimizationProfile(profile);
            if (profile_index < 0) {
                throw std::runtime_error("TensorRT addOptimizationProfile failed.");
            }
        }

        std::unique_ptr<nvinfer1::IHostMemory> serialized(
            builder->buildSerializedNetwork(*network, *config)
        );
        if (!serialized || serialized->size() == 0) {
            throw std::runtime_error("TensorRT buildSerializedNetwork failed.");
        }

        const fs::path out_path(engine_out_path);
        if (out_path.has_parent_path()) {
            fs::create_directories(out_path.parent_path());
        }

        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to create rebuilt engine file: " + out_path.string());
        }
        out.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        out.close();

        return out_path.string();
#else
        (void)onnx_path;
        (void)engine_out_path;
        (void)input_w;
        (void)input_h;
        throw std::runtime_error("TensorRT ONNX parser not enabled in this build.");
#endif
    }

    std::vector<TensorRTOutput> inferFromCHW(const std::vector<float>& input_fp32, int input_w, int input_h) {
        if (input_fp32.empty()) {
            return {};
        }
        nvinfer1::Dims4 in_shape(1, 3, input_h, input_w);
        if (!context_->setInputShape(input_name_.c_str(), in_shape)) {
            throw std::runtime_error("TensorRT setInputShape failed for " + input_name_);
        }

        const int nb = engine_->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            const char* n = engine_->getIOTensorName(i);
            if (n == nullptr) {
                continue;
            }
            const std::string name(n);
            const auto shape = context_->getTensorShape(name.c_str());
            const auto dtype = engine_->getTensorDataType(name.c_str());
            const auto mode = engine_->getTensorIOMode(name.c_str());

            const size_t vol = volumeFromDims(shape);
            if (vol == 0) {
                throw std::runtime_error("TensorRT tensor has invalid shape for " + name);
            }

            const size_t elem_size = dataTypeSize(dtype);
            if (elem_size == 0) {
                throw std::runtime_error("Unsupported TensorRT tensor dtype for " + name);
            }

            auto& b = buffers_[name];
            b.dtype = dtype;
            b.is_input = (mode == nvinfer1::TensorIOMode::kINPUT);
            b.shape.clear();
            for (int d = 0; d < shape.nbDims; ++d) {
                b.shape.push_back(shape.d[d]);
            }

            const size_t needed = vol * elem_size;
            if (b.bytes < needed || b.device_ptr == nullptr) {
                if (b.device_ptr != nullptr) {
                    checkCuda(cudaFree(b.device_ptr), "cudaFree(old tensor buffer)");
                }
                checkCuda(cudaMalloc(&b.device_ptr, needed), "cudaMalloc(tensor buffer)");
                b.bytes = needed;
            }

            if (!context_->setTensorAddress(name.c_str(), b.device_ptr)) {
                throw std::runtime_error("TensorRT setTensorAddress failed for " + name);
            }

            if (!b.is_input) {
                b.host_bytes.resize(needed);
            }
        }

        auto& in_buffer = buffers_.at(input_name_);
        const size_t expected_numel = static_cast<size_t>(3) * static_cast<size_t>(input_h) * static_cast<size_t>(input_w);
        if (input_fp32.size() != expected_numel) {
            throw std::runtime_error("TensorRT input size mismatch for " + input_name_);
        }

        if (in_buffer.dtype == nvinfer1::DataType::kFLOAT) {
            checkCuda(
                cudaMemcpyAsync(
                    in_buffer.device_ptr,
                    input_fp32.data(),
                    input_fp32.size() * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream_
                ),
                "cudaMemcpyAsync(input fp32)"
            );
        } else if (in_buffer.dtype == nvinfer1::DataType::kHALF) {
            std::vector<__half> input_fp16(input_fp32.size());
            for (size_t i = 0; i < input_fp32.size(); ++i) {
                input_fp16[i] = __float2half(input_fp32[i]);
            }
            checkCuda(
                cudaMemcpyAsync(
                    in_buffer.device_ptr,
                    input_fp16.data(),
                    input_fp16.size() * sizeof(__half),
                    cudaMemcpyHostToDevice,
                    stream_
                ),
                "cudaMemcpyAsync(input fp16)"
            );
        } else {
            throw std::runtime_error("Unsupported TensorRT input dtype for " + input_name_);
        }

        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 failed.");
        }

        for (const auto& out_name : output_names_) {
            auto& b = buffers_.at(out_name);
            checkCuda(
                cudaMemcpyAsync(
                    b.host_bytes.data(),
                    b.device_ptr,
                    b.bytes,
                    cudaMemcpyDeviceToHost,
                    stream_
                ),
                "cudaMemcpyAsync(output)"
            );
        }

        checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

        std::vector<TensorRTOutput> out;
        out.reserve(output_names_.size());

        for (const auto& out_name : output_names_) {
            const auto& b = buffers_.at(out_name);
            TensorRTOutput t;
            t.name = out_name;
            t.shape = b.shape;

            const size_t numel = b.bytes / dataTypeSize(b.dtype);
            t.data.resize(numel);

            if (b.dtype == nvinfer1::DataType::kFLOAT) {
                const float* src = reinterpret_cast<const float*>(b.host_bytes.data());
                std::copy(src, src + numel, t.data.begin());
            } else if (b.dtype == nvinfer1::DataType::kHALF) {
                const __half* src = reinterpret_cast<const __half*>(b.host_bytes.data());
                for (size_t i = 0; i < numel; ++i) {
                    t.data[i] = __half2float(src[i]);
                }
            } else if (b.dtype == nvinfer1::DataType::kINT32) {
                const int32_t* src = reinterpret_cast<const int32_t*>(b.host_bytes.data());
                for (size_t i = 0; i < numel; ++i) {
                    t.data[i] = static_cast<float>(src[i]);
                }
            } else if (b.dtype == nvinfer1::DataType::kINT8) {
                const int8_t* src = reinterpret_cast<const int8_t*>(b.host_bytes.data());
                for (size_t i = 0; i < numel; ++i) {
                    t.data[i] = static_cast<float>(src[i]);
                }
            } else {
                throw std::runtime_error("Unsupported TensorRT output dtype for " + out_name);
            }

            out.push_back(std::move(t));
        }

        return out;
    }

    std::vector<TensorRTOutput> infer(const cv::Mat& frame_bgr, int input_w, int input_h) {
        if (frame_bgr.empty()) {
            return {};
        }

        cv::Mat resized;
        cv::resize(frame_bgr, resized, cv::Size(input_w, input_h), 0, 0, cv::INTER_AREA);

        const int hw = input_h * input_w;
        std::vector<float> input_fp32(static_cast<size_t>(3 * hw));
        for (int y = 0; y < input_h; ++y) {
            const auto* row = resized.ptr<cv::Vec3b>(y);
            for (int x = 0; x < input_w; ++x) {
                const int idx = y * input_w + x;
                const cv::Vec3b px = row[x];
                input_fp32[idx] = static_cast<float>(px[2]) / 255.0F;
                input_fp32[hw + idx] = static_cast<float>(px[1]) / 255.0F;
                input_fp32[2 * hw + idx] = static_cast<float>(px[0]) / 255.0F;
            }
        }
        return inferFromCHW(input_fp32, input_w, input_h);
    }

private:
    struct TensorBuffer {
        void* device_ptr{nullptr};
        size_t bytes{0};
        bool is_input{false};
        nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
        std::vector<int64_t> shape;
        std::vector<uint8_t> host_bytes;
    };

    std::string engine_path_;
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_{nullptr};

    std::string input_name_;
    std::vector<std::string> output_names_;
    std::unordered_map<std::string, TensorBuffer> buffers_;
};

#endif

class YOLODetector {
public:
    explicit YOLODetector(
        const std::string& model_path,
        float confidence_threshold,
        int trail_length = 50,
        double approach_threshold = 1.1,
        double alert_duration = 1.5,
        double no_det_reset_sec = 1.5,
        const std::string& alert_message = "# ALERTA: APROXIMACAO DETECTADA"
    )
        : model_path_(model_path),
          confidence_threshold_(confidence_threshold),
          trail_length_(trail_length),
          approach_area_threshold_(approach_threshold),
          alert_duration_(alert_duration),
          no_det_reset_sec_(no_det_reset_sec),
          alert_message_(alert_message),
          tracker_(trail_length) {
        loadModel();
    }

    cv::Mat processFrame(const cv::Mat& frame) {
        if (frame.empty()) {
            return frame;
        }

        cv::Mat output = frame.clone();
        const std::vector<Detection> detections = infer(frame);

        const double now = nowSeconds();
        const bool has_detection = !detections.empty();

        float current_frame_max_area = 0.0F;
        for (const auto& d : detections) {
            const float area = static_cast<float>(d.box.area());
            current_frame_max_area = std::max(current_frame_max_area, area);
        }

        if (has_detection) {
            last_detection_time_ = now;
            if (global_max_area_ > 0.0F && current_frame_max_area > global_max_area_ * approach_area_threshold_) {
                last_approach_time_ = now;
            }
            global_max_area_ = std::max(global_max_area_, current_frame_max_area);
        } else if (last_detection_time_ > 0.0 && (now - last_detection_time_) > no_det_reset_sec_) {
            global_max_area_ = 0.0F;
        }

        const auto tracked = tracker_.update(detections);
        for (const auto& t : tracked) {
            cv::rectangle(output, t.det.box, cv::Scalar(0, 255, 0), 2);
            const int y_label = t.det.box.y > 20 ? t.det.box.y - 8 : t.det.box.y + 20;

            std::ostringstream lbl;
            lbl << "id:" << t.track_id << " " << std::fixed << std::setprecision(2) << t.det.confidence;
            cv::putText(output, lbl.str(), cv::Point(t.det.box.x, y_label), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

            if (t.trail.size() > 1) {
                std::vector<cv::Point> pts(t.trail.begin(), t.trail.end());
                cv::polylines(output, pts, false, t.color, 2);
            }
        }

        drawAlert(output, now);
        return output;
    }

private:
    struct TensorLayout {
        int channels{0};
        int anchors{0};
        bool channel_first{true};
    };

    template <typename Getter>
    std::vector<Detection> decodeDetections(
        int channels,
        int anchors,
        Getter&& getValue,
        int frame_w,
        int frame_h
    ) const {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        if (channels <= 0 || anchors <= 0) {
            return {};
        }

        // Case 1: tensor already in per-row format [N, 6] -> [x1,y1,x2,y2,conf,cls]
        if (channels >= 6 && channels <= 8) {
            for (int i = 0; i < anchors; ++i) {
                float x1 = getValue(0, i);
                float y1 = getValue(1, i);
                float x2 = getValue(2, i);
                float y2 = getValue(3, i);
                const float conf = getValue(4, i);
                const int cls = static_cast<int>(std::round(getValue(5, i)));

                if (conf < confidence_threshold_) {
                    continue;
                }

                // If format is xywh, convert.
                if (x2 <= x1 || y2 <= y1) {
                    const float cx = x1;
                    const float cy = y1;
                    const float w = std::max(0.0F, x2);
                    const float h = std::max(0.0F, y2);
                    x1 = cx - w * 0.5F;
                    y1 = cy - h * 0.5F;
                    x2 = cx + w * 0.5F;
                    y2 = cy + h * 0.5F;
                }

                float scale_x = 1.0F;
                float scale_y = 1.0F;

                const float max_coord = std::max(std::max(x1, x2), std::max(y1, y2));
                if (max_coord <= 1.5F) {
                    scale_x = static_cast<float>(frame_w);
                    scale_y = static_cast<float>(frame_h);
                } else if (max_coord <= static_cast<float>(input_w_ + 2)) {
                    scale_x = static_cast<float>(frame_w) / static_cast<float>(input_w_);
                    scale_y = static_cast<float>(frame_h) / static_cast<float>(input_h_);
                }

                int rx1 = static_cast<int>(std::round(x1 * scale_x));
                int ry1 = static_cast<int>(std::round(y1 * scale_y));
                int rx2 = static_cast<int>(std::round(x2 * scale_x));
                int ry2 = static_cast<int>(std::round(y2 * scale_y));

                rx1 = std::clamp(rx1, 0, frame_w - 1);
                ry1 = std::clamp(ry1, 0, frame_h - 1);
                rx2 = std::clamp(rx2, 0, frame_w - 1);
                ry2 = std::clamp(ry2, 0, frame_h - 1);

                if (rx2 <= rx1 || ry2 <= ry1) {
                    continue;
                }

                boxes.emplace_back(cv::Rect(cv::Point(rx1, ry1), cv::Point(rx2, ry2)));
                scores.push_back(conf);
                class_ids.push_back(cls);
            }
        } else {
            // Case 2: Ultralytics raw seg/det tensor [C, A] where C=4+nc(+mask)
            int num_classes = channels - 4 - 32;
            if (num_classes <= 0) {
                num_classes = channels - 4;
            }
            if (num_classes <= 0) {
                return {};
            }

            const float sx = static_cast<float>(frame_w) / static_cast<float>(input_w_);
            const float sy = static_cast<float>(frame_h) / static_cast<float>(input_h_);

            for (int a = 0; a < anchors; ++a) {
                const float cx = getValue(0, a);
                const float cy = getValue(1, a);
                const float w = getValue(2, a);
                const float h = getValue(3, a);

                float best_conf = -1.0F;
                int best_cls = 0;
                for (int c = 0; c < num_classes; ++c) {
                    const float score = getValue(4 + c, a);
                    if (score > best_conf) {
                        best_conf = score;
                        best_cls = c;
                    }
                }

                if (best_conf < confidence_threshold_) {
                    continue;
                }

                const float x1 = (cx - 0.5F * w) * sx;
                const float y1 = (cy - 0.5F * h) * sy;
                const float x2 = (cx + 0.5F * w) * sx;
                const float y2 = (cy + 0.5F * h) * sy;

                const int rx1 = std::clamp(static_cast<int>(std::round(x1)), 0, frame_w - 1);
                const int ry1 = std::clamp(static_cast<int>(std::round(y1)), 0, frame_h - 1);
                const int rx2 = std::clamp(static_cast<int>(std::round(x2)), 0, frame_w - 1);
                const int ry2 = std::clamp(static_cast<int>(std::round(y2)), 0, frame_h - 1);

                if (rx2 <= rx1 || ry2 <= ry1) {
                    continue;
                }

                boxes.emplace_back(cv::Rect(cv::Point(rx1, ry1), cv::Point(rx2, ry2)));
                scores.push_back(best_conf);
                class_ids.push_back(best_cls);
            }
        }

        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, confidence_threshold_, nms_threshold_, keep);

        std::vector<Detection> out;
        out.reserve(keep.size());
        for (int idx : keep) {
            Detection d;
            d.box = boxes[idx];
            d.confidence = scores[idx];
            d.class_id = class_ids[idx];
            out.push_back(d);
        }
        return out;
    }

    std::vector<Detection> decodeFromCvMat(const cv::Mat& pred, int frame_w, int frame_h) const {
        if (pred.empty()) {
            return {};
        }

        cv::Mat p32;
        if (pred.depth() == CV_32F) {
            p32 = pred;
        } else {
            pred.convertTo(p32, CV_32F);
        }

        TensorLayout layout;
        if (p32.dims == 3) {
            const int d1 = p32.size[1];
            const int d2 = p32.size[2];
            if (d1 <= d2) {
                layout.channels = d1;
                layout.anchors = d2;
                layout.channel_first = true;
            } else {
                layout.channels = d2;
                layout.anchors = d1;
                layout.channel_first = false;
            }
        } else if (p32.dims == 2) {
            const int rows = p32.size[0];
            const int cols = p32.size[1];
            layout.anchors = rows;
            layout.channels = cols;
            layout.channel_first = false;
        } else {
            return {};
        }

        const float* data = reinterpret_cast<const float*>(p32.data);
        auto getter = [&](int c, int a) -> float {
            if (layout.channel_first) {
                return data[static_cast<size_t>(c) * layout.anchors + a];
            }
            return data[static_cast<size_t>(a) * layout.channels + c];
        };

        return decodeDetections(layout.channels, layout.anchors, getter, frame_w, frame_h);
    }

#if defined(DEA_ENABLE_TENSORRT) && DEA_ENABLE_TENSORRT
    std::vector<Detection> decodeFromTensorRT(const std::vector<TensorRTOutput>& outputs, int frame_w, int frame_h) const {
        if (outputs.empty()) {
            return {};
        }

        const TensorRTOutput* best = nullptr;
        size_t best_score = 0;
        for (const auto& o : outputs) {
            if (o.shape.size() < 2 || o.data.empty()) {
                continue;
            }
            if (o.shape.size() == 3) {
                const auto d1 = static_cast<size_t>(std::max<int64_t>(1, o.shape[1]));
                const auto d2 = static_cast<size_t>(std::max<int64_t>(1, o.shape[2]));
                const size_t score = d1 * d2;
                if (score > best_score) {
                    best_score = score;
                    best = &o;
                }
            } else if (o.shape.size() == 2) {
                const auto d0 = static_cast<size_t>(std::max<int64_t>(1, o.shape[0]));
                const auto d1 = static_cast<size_t>(std::max<int64_t>(1, o.shape[1]));
                const size_t score = d0 * d1;
                if (score > best_score) {
                    best_score = score;
                    best = &o;
                }
            }
        }

        if (best == nullptr) {
            return {};
        }

        int channels = 0;
        int anchors = 0;
        bool channel_first = true;

        if (best->shape.size() == 3) {
            const int d1 = static_cast<int>(best->shape[1]);
            const int d2 = static_cast<int>(best->shape[2]);
            if (d1 <= d2) {
                channels = d1;
                anchors = d2;
                channel_first = true;
            } else {
                channels = d2;
                anchors = d1;
                channel_first = false;
            }
        } else if (best->shape.size() == 2) {
            anchors = static_cast<int>(best->shape[0]);
            channels = static_cast<int>(best->shape[1]);
            channel_first = false;
        } else {
            return {};
        }

        auto getter = [&](int c, int a) -> float {
            if (channel_first) {
                return best->data[static_cast<size_t>(c) * anchors + a];
            }
            return best->data[static_cast<size_t>(a) * channels + c];
        };

        return decodeDetections(channels, anchors, getter, frame_w, frame_h);
    }
#endif

    std::vector<Detection> infer(const cv::Mat& frame) {
#if defined(DEA_ENABLE_TENSORRT) && DEA_ENABLE_TENSORRT
        if (use_tensorrt_ && trt_engine_) {
            const auto trt_outputs = trt_engine_->infer(frame, input_w_, input_h_);
            return decodeFromTensorRT(trt_outputs, frame.cols, frame.rows);
        }
#endif
        if (!use_dnn_) {
            return {};
        }

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_AREA);

        cv::Mat blob = cv::dnn::blobFromImage(
            resized,
            1.0 / 255.0,
            cv::Size(input_w_, input_h_),
            cv::Scalar(),
            true,
            false,
            CV_32F
        );

        dnn_net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        dnn_net_.forward(outputs, dnn_output_names_);

        const cv::Mat* pred = nullptr;
        size_t best_total = 0;
        for (auto& o : outputs) {
            if (o.empty()) {
                continue;
            }
            if (o.dims == 3 || o.dims == 2) {
                if (o.total() > best_total) {
                    best_total = o.total();
                    pred = &o;
                }
            }
        }

        if (pred == nullptr) {
            return {};
        }

        return decodeFromCvMat(*pred, frame.cols, frame.rows);
    }

    void loadDnnModel(const std::string& onnx_path) {
        dnn_net_ = cv::dnn::readNetFromONNX(onnx_path);
        if (dnn_net_.empty()) {
            throw std::runtime_error("Failed to load YOLO ONNX: " + onnx_path);
        }

        try {
            if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                dnn_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                dnn_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            } else {
                dnn_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                dnn_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        } catch (...) {
            dnn_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            dnn_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        dnn_output_names_ = dnn_net_.getUnconnectedOutLayersNames();
        use_dnn_ = true;
        std::cout << "✓ YOLO ONNX carregado: " << onnx_path << '\n';
    }

    void loadModel() {
        const fs::path p(model_path_);
        const std::string ext = toLower(p.extension().string());

        if (ext == ".engine") {
#if defined(DEA_ENABLE_TENSORRT) && DEA_ENABLE_TENSORRT
            try {
                trt_engine_ = std::make_unique<TensorRTEngine>(model_path_);
                use_tensorrt_ = true;
                std::cout << "✓ YOLO TensorRT carregado (.engine)" << '\n';
                return;
            } catch (const std::exception& e) {
                std::cerr << "Aviso: falha ao carregar .engine: " << e.what() << '\n';
                const fs::path onnx_fallback = p.parent_path() / (p.stem().string() + ".onnx");
                if (fs::exists(onnx_fallback)) {
                    const fs::path rebuilt_engine = p.parent_path() / (p.stem().string() + "_rebuilt.engine");
#if defined(DEA_ENABLE_TRT_ONNX_PARSER) && DEA_ENABLE_TRT_ONNX_PARSER
                    if (fs::exists(rebuilt_engine)) {
                        try {
                            trt_engine_ = std::make_unique<TensorRTEngine>(rebuilt_engine.string());
                            use_tensorrt_ = true;
                            std::cout << "✓ YOLO TensorRT rebuilt carregado: " << rebuilt_engine << '\n';
                            return;
                        } catch (const std::exception& rebuilt_load_err) {
                            std::cerr << "Aviso: arquivo rebuilt existente mas invalido: " << rebuilt_load_err.what() << '\n';
                        }
                    }

                    std::cerr << "Tentando rebuild TensorRT a partir do ONNX: " << onnx_fallback << '\n';
                    try {
                        const std::string rebuilt_path = TensorRTEngine::buildEngineFromOnnx(
                            onnx_fallback.string(),
                            rebuilt_engine.string(),
                            input_w_,
                            input_h_
                        );
                        trt_engine_ = std::make_unique<TensorRTEngine>(rebuilt_path);
                        use_tensorrt_ = true;
                        std::cout << "✓ YOLO TensorRT rebuilt carregado: " << rebuilt_path << '\n';
                        return;
                    } catch (const std::exception& trt_build_err) {
                        std::cerr << "Aviso: rebuild TensorRT do ONNX falhou: " << trt_build_err.what() << '\n';
                    }
#endif
                    std::cerr << "Tentando fallback ONNX via OpenCV DNN: " << onnx_fallback << '\n';
                    loadDnnModel(onnx_fallback.string());
                    return;
                }
                throw;
            }
#else
            throw std::runtime_error("Build sem TensorRT. Recompile com TensorRT para usar .engine, ou use .onnx.");
#endif
        }

        if (ext == ".onnx") {
            loadDnnModel(model_path_);
            return;
        }

        throw std::runtime_error("Formato YOLO nao suportado: " + ext + " (use .engine ou .onnx)");
    }

    void drawAlert(cv::Mat& frame, double now) const {
        if (now >= last_approach_time_ + alert_duration_) {
            return;
        }

        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(alert_message_, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
        const int pad = 5;
        const int x1 = 15 - pad;
        const int y1 = 80 - text_size.height - pad;
        const int x2 = 15 + text_size.width + pad;
        const int y2 = 80 + baseline + pad;

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, alert_message_, cv::Point(15, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    std::string model_path_;
    float confidence_threshold_{0.6F};
    int trail_length_{50};
    double approach_area_threshold_{1.1};
    double alert_duration_{1.5};
    double no_det_reset_sec_{1.5};
    std::string alert_message_;

    int input_w_{640};
    int input_h_{640};
    float nms_threshold_{0.45F};

    float global_max_area_{0.0F};
    double last_approach_time_{0.0};
    double last_detection_time_{0.0};

    SimpleTracker tracker_;

    bool use_dnn_{false};
    cv::dnn::Net dnn_net_;
    std::vector<cv::String> dnn_output_names_;

#if defined(DEA_ENABLE_TENSORRT) && DEA_ENABLE_TENSORRT
    bool use_tensorrt_{false};
    std::unique_ptr<TensorRTEngine> trt_engine_;
#endif
};

class SkySegmentation {
public:
    SkySegmentation(
        const std::string& model_path,
        cv::Size input_size = cv::Size(320, 320),
        int update_interval = 1,
        int sample_area_size = 30,
        double sky_upper_threshold = 0.75,
        double sky_lower_threshold = 0.25,
        int binary_threshold = 128
    )
        : model_path_(model_path),
          input_size_(input_size),
          update_interval_(std::max(1, update_interval)),
          sample_area_size_(sample_area_size),
          sky_upper_threshold_(sky_upper_threshold),
          sky_lower_threshold_(sky_lower_threshold),
          binary_threshold_(binary_threshold) {
        loadModel();
    }

    cv::Mat processFrame(const cv::Mat& frame) {
        if (frame.empty()) {
            return frame;
        }
        if (!valid_) {
            return frame.clone();
        }

        frame_count_ += 1;
        const bool should_update = ((frame_count_ - 1) % update_interval_) == 0;

        if (should_update || last_mask_.empty()) {
            const cv::Mat mask_gray = runInference(frame);
            cv::Mat binary_mask;
            cv::threshold(mask_gray, binary_mask, binary_threshold_, 255, cv::THRESH_BINARY);

            analyzeFlightDirection(binary_mask);

            cv::cvtColor(binary_mask, last_mask_, cv::COLOR_GRAY2BGR);
        }

        cv::Mat display = last_mask_.clone();
        drawFlightStatus(display);
        return display;
    }

private:
    void loadModel() {
        if (!fs::exists(model_path_)) {
            throw std::runtime_error("SkySeg model not found: " + model_path_);
        }

        const fs::path model_p(model_path_);
        const std::string ext = toLower(model_p.extension().string());

#if defined(DEA_ENABLE_TENSORRT) && DEA_ENABLE_TENSORRT
        auto tryLoadTrtEngine = [&](const fs::path& engine_path) -> bool {
            try {
                trt_engine_ = std::make_unique<TensorRTEngine>(engine_path.string());
                use_tensorrt_ = true;
                std::cout << "✓ SkySeg TensorRT carregado: " << engine_path << '\n';
                return true;
            } catch (const std::exception& e) {
                std::cerr << "Aviso: falha ao carregar SkySeg TensorRT (" << engine_path << "): " << e.what() << '\n';
                return false;
            }
        };

        if (ext == ".engine") {
            valid_ = tryLoadTrtEngine(model_p);
            if (valid_) {
                return;
            }
        }

        if (ext == ".onnx") {
            fs::path cache_dir = projectRootPath() / "trt_cache";
            fs::create_directories(cache_dir);
            const fs::path cached_engine = cache_dir / (model_p.stem().string() + "_trt.engine");

            if (fs::exists(cached_engine) && tryLoadTrtEngine(cached_engine)) {
                valid_ = true;
                return;
            }

#if defined(DEA_ENABLE_TRT_ONNX_PARSER) && DEA_ENABLE_TRT_ONNX_PARSER
            try {
                const std::string rebuilt = TensorRTEngine::buildEngineFromOnnx(
                    model_p.string(),
                    cached_engine.string(),
                    input_size_.width,
                    input_size_.height
                );
                if (tryLoadTrtEngine(rebuilt)) {
                    valid_ = true;
                    return;
                }
            } catch (const std::exception& e) {
                std::cerr << "Aviso: build SkySeg TensorRT falhou: " << e.what() << '\n';
            }
#endif
        }
#endif

        if (ext != ".onnx") {
            throw std::runtime_error("SkySeg sem TensorRT requer modelo ONNX.");
        }

        net_ = cv::dnn::readNetFromONNX(model_path_);
        if (net_.empty()) {
            throw std::runtime_error("Failed to load SkySeg ONNX: " + model_path_);
        }

        try {
            if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            } else {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        } catch (...) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        use_tensorrt_ = false;
        valid_ = true;
        std::cout << "✓ SkySeg ONNX (OpenCV DNN) carregado: " << model_path_ << '\n';
    }

    cv::Mat postprocessMaskFloat(const cv::Mat& mask_float, int original_width, int original_height) {
        double minv = 0.0;
        double maxv = 0.0;
        cv::minMaxLoc(mask_float, &minv, &maxv);

        cv::Mat mask_norm;
        if (maxv > minv) {
            mask_norm = (mask_float - minv) / (maxv - minv);
        } else {
            mask_norm = cv::Mat::zeros(mask_float.size(), CV_32F);
        }

        cv::Mat mask_u8;
        mask_norm.convertTo(mask_u8, CV_8U, 255.0);

        cv::Mat resized_mask;
        cv::resize(mask_u8, resized_mask, cv::Size(original_width, original_height), 0, 0, cv::INTER_NEAREST);
        return resized_mask;
    }

    cv::Mat runInferenceDnn(const cv::Mat& image_bgr) {
        const int original_height = image_bgr.rows;
        const int original_width = image_bgr.cols;

        cv::Mat resized;
        cv::resize(image_bgr, resized, input_size_, 0, 0, cv::INTER_AREA);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::Mat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(float_img, channels);
        const std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
        const std::array<float, 3> stdv{0.229F, 0.224F, 0.225F};
        for (int c = 0; c < 3; ++c) {
            channels[c] = (channels[c] - mean[c]) / stdv[c];
        }
        cv::merge(channels, float_img);

        cv::Mat blob = cv::dnn::blobFromImage(
            float_img,
            1.0,
            input_size_,
            cv::Scalar(),
            false,
            false,
            CV_32F
        );

        net_.setInput(blob);
        cv::Mat out = net_.forward();
        if (out.empty()) {
            throw std::runtime_error("SkySeg inference returned empty output.");
        }

        cv::Mat mask_float;
        if (out.dims == 4) {
            const int h = out.size[2];
            const int w = out.size[3];
            cv::Mat view(h, w, CV_32F, out.ptr<float>());
            mask_float = view.clone();
        } else if (out.dims == 3) {
            const int h = out.size[1];
            const int w = out.size[2];
            cv::Mat view(h, w, CV_32F, out.ptr<float>());
            mask_float = view.clone();
        } else if (out.dims == 2) {
            out.convertTo(mask_float, CV_32F);
        } else {
            throw std::runtime_error("SkySeg unexpected output dims.");
        }

        return postprocessMaskFloat(mask_float, original_width, original_height);
    }

#if defined(DEA_ENABLE_TENSORRT) && DEA_ENABLE_TENSORRT
    cv::Mat runInferenceTensorRT(const cv::Mat& image_bgr) {
        if (!trt_engine_) {
            throw std::runtime_error("SkySeg TensorRT engine not initialized.");
        }

        const int original_height = image_bgr.rows;
        const int original_width = image_bgr.cols;
        const int in_w = input_size_.width;
        const int in_h = input_size_.height;
        const int hw = in_w * in_h;

        cv::Mat resized;
        cv::resize(image_bgr, resized, input_size_, 0, 0, cv::INTER_AREA);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        const std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
        const std::array<float, 3> stdv{0.229F, 0.224F, 0.225F};
        for (int c = 0; c < 3; ++c) {
            channels[c] = (channels[c] - mean[c]) / stdv[c];
        }

        std::vector<float> input_chw(static_cast<size_t>(3 * hw));
        for (int c = 0; c < 3; ++c) {
            const cv::Mat& ch = channels[c];
            for (int y = 0; y < in_h; ++y) {
                const float* row = ch.ptr<float>(y);
                std::copy(row, row + in_w, input_chw.begin() + static_cast<size_t>(c * hw + y * in_w));
            }
        }

        const auto outputs = trt_engine_->inferFromCHW(input_chw, in_w, in_h);
        if (outputs.empty()) {
            throw std::runtime_error("SkySeg TensorRT returned no outputs.");
        }

        const TensorRTOutput* best = nullptr;
        for (const auto& o : outputs) {
            if (o.data.empty()) {
                continue;
            }
            if (best == nullptr || o.data.size() > best->data.size()) {
                best = &o;
            }
        }
        if (!best) {
            throw std::runtime_error("SkySeg TensorRT outputs are empty.");
        }

        int h = in_h;
        int w = in_w;
        if (best->shape.size() >= 4) {
            h = static_cast<int>(best->shape[best->shape.size() - 2]);
            w = static_cast<int>(best->shape[best->shape.size() - 1]);
        } else if (best->shape.size() == 3) {
            h = static_cast<int>(best->shape[1]);
            w = static_cast<int>(best->shape[2]);
        } else if (best->shape.size() == 2) {
            h = static_cast<int>(best->shape[0]);
            w = static_cast<int>(best->shape[1]);
        }
        if (h <= 0 || w <= 0) {
            h = in_h;
            w = in_w;
        }

        const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w);
        if (best->data.size() < plane) {
            throw std::runtime_error("SkySeg TensorRT output size is invalid.");
        }

        cv::Mat mask_float(h, w, CV_32F);
        std::memcpy(mask_float.data, best->data.data(), plane * sizeof(float));
        return postprocessMaskFloat(mask_float, original_width, original_height);
    }
#endif

    cv::Mat runInference(const cv::Mat& image_bgr) {
#if defined(DEA_ENABLE_TENSORRT) && DEA_ENABLE_TENSORRT
        if (use_tensorrt_ && trt_engine_) {
            return runInferenceTensorRT(image_bgr);
        }
#endif
        return runInferenceDnn(image_bgr);
    }

    void analyzeFlightDirection(const cv::Mat& binary_mask) {
        const int h = binary_mask.rows;
        const int w = binary_mask.cols;

        const int cx = w / 2;
        const int cy = h / 2;
        const int half = sample_area_size_ / 2;

        const int x0 = std::max(0, cx - half);
        const int y0 = std::max(0, cy - half);
        const int x1 = std::min(w, cx + half);
        const int y1 = std::min(h, cy + half);

        if (x1 <= x0 || y1 <= y0) {
            last_flight_status_ = "DESCONHECIDO";
            last_sky_ratio_ = 0.0;
            last_roi_ = cv::Rect();
            last_status_color_ = cv::Scalar(128, 128, 128);
            return;
        }

        const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
        const cv::Mat center_roi = binary_mask(roi);

        const double mean_val = cv::mean(center_roi)[0];
        const double sky_ratio = mean_val / 255.0;

        last_sky_ratio_ = sky_ratio;
        last_roi_ = roi;

        if (sky_ratio > sky_upper_threshold_) {
            last_flight_status_ = "SUBINDO";
            last_status_color_ = cv::Scalar(0, 255, 255);
        } else if (sky_ratio < sky_lower_threshold_) {
            last_flight_status_ = "DESCENDO";
            last_status_color_ = cv::Scalar(255, 0, 0);
        } else {
            last_flight_status_ = "NIVELADO";
            last_status_color_ = cv::Scalar(0, 255, 0);
        }
    }

    void drawCenterCross(cv::Mat& img, int size, const cv::Scalar& color, int thickness) const {
        const int cx = img.cols / 2;
        const int cy = img.rows / 2;
        cv::line(img, cv::Point(cx - size, cy), cv::Point(cx + size, cy), color, thickness);
        cv::line(img, cv::Point(cx, cy - size), cv::Point(cx, cy + size), color, thickness);
    }

    void drawFlightStatus(cv::Mat& frame) const {
        if (frame.empty()) {
            return;
        }

        if (last_roi_.area() > 0) {
            cv::rectangle(frame, last_roi_, last_status_color_, 3);
        }
        drawCenterCross(frame, 15, last_status_color_, 2);

        std::ostringstream st;
        st << "VOO: " << last_flight_status_;

        std::ostringstream rt;
        rt << std::fixed << std::setprecision(1) << (last_sky_ratio_ * 100.0);
        const std::string ratio_text = "CEU: " + rt.str() + "%";

        const int text_x = std::max(10, frame.cols - 220);
        cv::putText(frame, st.str(), cv::Point(text_x, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, last_status_color_, 2);
        cv::putText(frame, ratio_text, cv::Point(text_x, 55), cv::FONT_HERSHEY_SIMPLEX, 0.6, last_status_color_, 2);
    }

    std::string model_path_;
    cv::dnn::Net net_;
    bool valid_{false};
    bool use_tensorrt_{false};

#if defined(DEA_ENABLE_TENSORRT) && DEA_ENABLE_TENSORRT
    std::unique_ptr<TensorRTEngine> trt_engine_;
#endif

    cv::Size input_size_{320, 320};
    int update_interval_{1};
    int sample_area_size_{30};
    double sky_upper_threshold_{0.75};
    double sky_lower_threshold_{0.25};
    int binary_threshold_{128};

    int frame_count_{0};
    cv::Mat last_mask_;
    std::string last_flight_status_{"DESCONHECIDO"};
    double last_sky_ratio_{0.0};
    cv::Rect last_roi_;
    cv::Scalar last_status_color_{128, 128, 128};
};

class OpticalFlowModule {
public:
    OpticalFlowModule(int clusters, double fps, cv::Size processing_size, bool flow_gpu_requested)
        : number_clusters_(std::max(2, clusters)),
          fps_(fps),
          processing_size_(processing_size),
          flow_gpu_requested_(flow_gpu_requested) {
#if defined(DEA_ENABLE_VPI) && DEA_ENABLE_VPI
        if (flow_gpu_requested_) {
            use_vpi_gpu_ = true;
            std::cout << "[flow] --flow-gpu habilitado: tentando VPI OpticalFlowPyrLK (CUDA)." << '\n';
        }
#else
        if (flow_gpu_requested_) {
            std::cout << "[flow] --flow-gpu solicitado, mas build sem VPI. Usando CPU." << '\n';
        }
#endif

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 255);
        const int n_colors = std::max(number_clusters_, 10);
        colors_.reserve(n_colors);
        for (int i = 0; i < n_colors; ++i) {
            colors_.emplace_back(dist(rng), dist(rng), dist(rng));
        }
    }

    ~OpticalFlowModule() {
#if defined(DEA_ENABLE_VPI) && DEA_ENABLE_VPI
        releaseVpi();
#endif
    }

    cv::Mat processFrame(const cv::Mat& frame) {
        if (frame.empty()) {
            return frame;
        }

        cv::Mat frame_resized;
        if (frame.size() == processing_size_) {
            frame_resized = frame;
        } else {
            cv::resize(frame, frame_resized, processing_size_, 0, 0, cv::INTER_AREA);
        }

#if defined(DEA_ENABLE_VPI) && DEA_ENABLE_VPI
        if (use_vpi_gpu_) {
            try {
                return processFrameVpi(frame_resized);
            } catch (const std::exception& e) {
                use_vpi_gpu_ = false;
                releaseVpi();
                std::cerr << "[flow] fallback para CPU apos falha VPI: " << e.what() << '\n';
                initializeTrackingCpu(frame_resized);
                return frame_resized;
            }
        }
#endif
        return processFrameCpu(frame_resized);
    }

private:
    static float pointNorm(const cv::Point2f& p) {
        return std::sqrt(p.x * p.x + p.y * p.y);
    }

    static cv::Mat buildBinaryMask(const cv::Mat& detection_mask) {
        cv::Mat binary;
        cv::threshold(detection_mask, binary, 127, 255, cv::THRESH_BINARY);
        return binary;
    }

    int createTrackedPoint(const cv::Point2f& pos) {
        const int id = next_point_id_++;
        tracked_points_[id] = pos;
        point_paths_[id] = {};
        point_clusters_[id] = 0;
        return id;
    }

    void eraseTrackedPoint(int id) {
        tracked_points_.erase(id);
        point_paths_.erase(id);
        point_clusters_.erase(id);
    }

    void initializeDetectionMaskIfNeeded(const cv::Size& gray_size) {
        if (detection_mask_.empty() || detection_mask_.size() != gray_size) {
            detection_mask_ = cv::Mat(gray_size, CV_8U, cv::Scalar(255));
        }
    }

    bool runFuzzyCMeans(
        const cv::Mat& data,
        int clusters,
        float fuzziness,
        float error,
        int max_iter,
        const cv::Mat& init_u,
        cv::Mat& centers_out,
        cv::Mat& membership_out
    ) {
        if (data.empty() || clusters <= 0 || data.rows < clusters) {
            return false;
        }

        const int n_points = data.rows;
        const int dims = data.cols;
        cv::Mat u;

        if (!init_u.empty() && init_u.rows == clusters && init_u.cols == n_points && init_u.type() == CV_32F) {
            u = init_u.clone();
        } else {
            u = cv::Mat(clusters, n_points, CV_32F);
            cv::RNG rng(42);
            for (int j = 0; j < n_points; ++j) {
                float sum = 0.0F;
                for (int i = 0; i < clusters; ++i) {
                    const float v = rng.uniform(0.001F, 1.0F);
                    u.at<float>(i, j) = v;
                    sum += v;
                }
                if (sum <= 0.0F) {
                    sum = 1.0F;
                }
                for (int i = 0; i < clusters; ++i) {
                    u.at<float>(i, j) /= sum;
                }
            }
        }

        cv::Mat centers(clusters, dims, CV_32F, cv::Scalar(0));
        cv::Mat prev_u = u.clone();
        const float exp = 2.0F / (fuzziness - 1.0F);

        for (int iter = 0; iter < max_iter; ++iter) {
            for (int i = 0; i < clusters; ++i) {
                float denom = 0.0F;
                cv::Mat num(1, dims, CV_32F, cv::Scalar(0));
                for (int j = 0; j < n_points; ++j) {
                    const float uij = u.at<float>(i, j);
                    const float w = std::pow(std::max(0.0F, uij), fuzziness);
                    denom += w;
                    for (int d = 0; d < dims; ++d) {
                        num.at<float>(0, d) += w * data.at<float>(j, d);
                    }
                }
                if (denom <= 1e-9F) {
                    continue;
                }
                for (int d = 0; d < dims; ++d) {
                    centers.at<float>(i, d) = num.at<float>(0, d) / denom;
                }
            }

            for (int j = 0; j < n_points; ++j) {
                std::vector<float> dists(static_cast<size_t>(clusters), 0.0F);
                int zero_idx = -1;
                for (int i = 0; i < clusters; ++i) {
                    float accum = 0.0F;
                    for (int d = 0; d < dims; ++d) {
                        const float diff = data.at<float>(j, d) - centers.at<float>(i, d);
                        accum += diff * diff;
                    }
                    dists[static_cast<size_t>(i)] = std::sqrt(accum);
                    if (dists[static_cast<size_t>(i)] <= 1e-9F) {
                        zero_idx = i;
                    }
                }

                if (zero_idx >= 0) {
                    for (int i = 0; i < clusters; ++i) {
                        u.at<float>(i, j) = (i == zero_idx) ? 1.0F : 0.0F;
                    }
                    continue;
                }

                for (int i = 0; i < clusters; ++i) {
                    float denom = 0.0F;
                    const float dij = std::max(dists[static_cast<size_t>(i)], 1e-9F);
                    for (int k = 0; k < clusters; ++k) {
                        const float dkj = std::max(dists[static_cast<size_t>(k)], 1e-9F);
                        denom += std::pow(dij / dkj, exp);
                    }
                    u.at<float>(i, j) = (denom <= 1e-9F) ? (1.0F / static_cast<float>(clusters)) : (1.0F / denom);
                }
            }

            cv::Mat diff;
            cv::absdiff(u, prev_u, diff);
            double max_diff = 0.0;
            cv::minMaxLoc(diff, nullptr, &max_diff);
            if (max_diff < static_cast<double>(error)) {
                break;
            }
            prev_u = u.clone();
        }

        centers_out = centers;
        membership_out = u;
        return true;
    }

    cv::Mat processFrameCpu(const cv::Mat& frame_resized) {
        if (old_gray_.empty() || tracked_points_.empty()) {
            initializeTrackingCpu(frame_resized);
            return frame_resized;
        }

        cv::Mat gray;
        cv::cvtColor(frame_resized, gray, cv::COLOR_BGR2GRAY);

        std::vector<int> point_ids;
        point_ids.reserve(tracked_points_.size());
        for (const auto& kv : tracked_points_) {
            point_ids.push_back(kv.first);
        }

        std::vector<cv::Point2f> p0;
        p0.reserve(point_ids.size());
        for (int id : point_ids) {
            p0.push_back(tracked_points_[id]);
        }

        std::vector<cv::Point2f> p1;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(old_gray_, gray, p0, p1, status, err, cv::Size(15, 15), 2, lk_criteria_);

        std::vector<int> good_ids;
        std::vector<cv::Point2f> good_new;
        std::vector<cv::Point2f> good_old;
        good_ids.reserve(point_ids.size());
        good_new.reserve(point_ids.size());
        good_old.reserve(point_ids.size());

        for (size_t i = 0; i < point_ids.size(); ++i) {
            if (i < status.size() && status[i] == 1 && i < p1.size()) {
                good_ids.push_back(point_ids[i]);
                good_new.push_back(p1[i]);
                good_old.push_back(p0[i]);
            }
        }

        for (int id : point_ids) {
            if (std::find(good_ids.begin(), good_ids.end(), id) == good_ids.end()) {
                eraseTrackedPoint(id);
            }
        }

        if (good_new.empty()) {
            initializeTrackingCpu(frame_resized);
            return frame_resized;
        }

        std::vector<cv::Point2f> uvs(good_new.size());
        for (size_t i = 0; i < good_new.size(); ++i) {
            uvs[i] = (good_new[i] - good_old[i]) * static_cast<float>(fps_);
        }

        for (size_t i = 0; i < good_ids.size(); ++i) {
            const int id = good_ids[i];
            tracked_points_[id] = good_new[i];
            auto& path = point_paths_[id];
            path.push_back(good_new[i]);
            while (static_cast<int>(path.size()) > max_path_length_) {
                path.pop_front();
            }
        }

        std::vector<int> valid_ids;
        std::vector<int> invalid_ids;
        valid_ids.reserve(good_ids.size());
        invalid_ids.reserve(good_ids.size());

        constexpr int min_frames_check = 5;
        constexpr float inconsistency_threshold = 0.85F;
        initializeDetectionMaskIfNeeded(gray.size());

        for (size_t i = 0; i < good_ids.size(); ++i) {
            const int id = good_ids[i];
            bool is_valid = true;
            auto it_path = point_paths_.find(id);
            if (it_path != point_paths_.end() && static_cast<int>(it_path->second.size()) >= min_frames_check) {
                const auto& path = it_path->second;
                std::vector<cv::Point2f> recent_vels;
                recent_vels.reserve(static_cast<size_t>(min_frames_check));

                const int start = static_cast<int>(path.size()) - min_frames_check;
                for (int j = start; j < static_cast<int>(path.size()) - 1; ++j) {
                    recent_vels.push_back((path[j + 1] - path[j]) * static_cast<float>(fps_));
                }

                if (recent_vels.size() >= 2) {
                    int invalid_transitions = 0;
                    int total_transitions = 0;
                    for (size_t j = 0; j + 1 < recent_vels.size(); ++j) {
                        const cv::Point2f vel1 = recent_vels[j];
                        const cv::Point2f vel2 = recent_vels[j + 1];
                        const float mag1 = pointNorm(vel1);
                        const float mag2 = pointNorm(vel2);
                        if (mag1 > 2.0F && mag2 > 2.0F) {
                            total_transitions += 1;
                            const float dot = vel1.x * vel2.x + vel1.y * vel2.y;
                            float cos_angle = dot / std::max(1e-6F, mag1 * mag2);
                            cos_angle = std::clamp(cos_angle, -1.0F, 1.0F);
                            const float angle = std::acos(cos_angle);
                            if (angle > static_cast<float>(CV_PI) / 2.0F) {
                                invalid_transitions += 1;
                                continue;
                            }
                            const float mag_ratio = std::max(mag1, mag2) / (std::min(mag1, mag2) + 1e-6F);
                            if (mag_ratio > 3.0F) {
                                invalid_transitions += 1;
                            }
                        }
                    }

                    if (total_transitions > 0) {
                        const float inconsistency = static_cast<float>(invalid_transitions) / static_cast<float>(total_transitions);
                        if (inconsistency >= inconsistency_threshold) {
                            is_valid = false;
                        }
                    }
                }
            }

            if (is_valid) {
                valid_ids.push_back(id);
            } else {
                invalid_ids.push_back(id);
                const cv::Point2f pos = good_new[i];
                cv::circle(detection_mask_, cv::Point(static_cast<int>(pos.x), static_cast<int>(pos.y)), invalid_region_radius_, cv::Scalar(0), cv::FILLED);
            }
        }

        for (int id : invalid_ids) {
            eraseTrackedPoint(id);
        }

        if (valid_ids.size() != good_ids.size()) {
            std::vector<cv::Point2f> filtered_new;
            std::vector<cv::Point2f> filtered_old;
            std::vector<cv::Point2f> filtered_uvs;
            filtered_new.reserve(valid_ids.size());
            filtered_old.reserve(valid_ids.size());
            filtered_uvs.reserve(valid_ids.size());

            for (size_t i = 0; i < good_ids.size(); ++i) {
                if (std::find(valid_ids.begin(), valid_ids.end(), good_ids[i]) != valid_ids.end()) {
                    filtered_new.push_back(good_new[i]);
                    filtered_old.push_back(good_old[i]);
                    filtered_uvs.push_back(uvs[i]);
                }
            }
            good_ids = std::move(valid_ids);
            good_new = std::move(filtered_new);
            good_old = std::move(filtered_old);
            uvs = std::move(filtered_uvs);

            if (good_new.empty()) {
                initializeTrackingCpu(frame_resized);
                return frame_resized;
            }
        }

        detection_mask_.convertTo(detection_mask_, CV_16S);
        detection_mask_ += mask_recovery_rate_;
        cv::threshold(detection_mask_, detection_mask_, 255, 255, cv::THRESH_TRUNC);
        detection_mask_.convertTo(detection_mask_, CV_8U);

        cv::Mat blended = renderClusteredFlow(frame_resized, good_ids, good_new, uvs);

        old_gray_ = gray;

        if (frame_iter_ >= std::max(1, static_cast<int>(0.5 * fps_)) - 1) {
            std::vector<cv::Point2f> new_features;
            cv::Mat binary_mask = buildBinaryMask(detection_mask_);
            cv::goodFeaturesToTrack(old_gray_, new_features, max_points_, 0.3, 7.0, binary_mask, 7, false, 0.04);

            std::vector<cv::Point2f> tmp_points = good_new;
            int available_slots = std::max(0, max_points_ - static_cast<int>(tracked_points_.size()));

            for (const auto& npt : new_features) {
                if (available_slots <= 0) {
                    break;
                }

                bool add = true;
                for (const auto& ept : tmp_points) {
                    const cv::Point2f d = npt - ept;
                    if ((d.x * d.x + d.y * d.y) < 16.0F) {
                        add = false;
                        break;
                    }
                }
                if (add) {
                    createTrackedPoint(npt);
                    tmp_points.push_back(npt);
                    available_slots -= 1;
                }
            }
            frame_iter_ = -1;
        }

        frame_iter_ += 1;
        return blended;
    }

    void initializeTrackingCpu(const cv::Mat& frame_resized) {
        cv::cvtColor(frame_resized, old_gray_, cv::COLOR_BGR2GRAY);

        initializeDetectionMaskIfNeeded(old_gray_.size());
        cv::Mat binary_mask = buildBinaryMask(detection_mask_);

        std::vector<cv::Point2f> points;
        cv::goodFeaturesToTrack(old_gray_, points, max_points_, 0.3, 7.0, binary_mask, 7, false, 0.04);

        tracked_points_.clear();
        point_paths_.clear();
        point_clusters_.clear();
        for (const auto& p : points) {
            createTrackedPoint(p);
        }

        frame_iter_ = 0;
    }

    cv::Mat renderClusteredFlow(
        const cv::Mat& frame_resized,
        const std::vector<int>& point_ids,
        const std::vector<cv::Point2f>& good_new,
        const std::vector<cv::Point2f>& uvs
    ) {
        std::vector<cv::Point2f> path_vectors(good_new.size(), cv::Point2f(0.0F, 0.0F));
        for (size_t i = 0; i < point_ids.size() && i < good_new.size(); ++i) {
            const auto it = point_paths_.find(point_ids[i]);
            if (it == point_paths_.end() || it->second.empty()) {
                continue;
            }
            cv::Point2f sum(0.0F, 0.0F);
            for (const auto& p : it->second) {
                sum += p;
            }
            path_vectors[i] = sum * (1.0F / static_cast<float>(it->second.size()));
        }

        std::vector<int> labels(good_new.size(), 0);
        cv::Mat centers;
        cv::Mat membership;

        if (static_cast<int>(good_new.size()) >= number_clusters_) {
            cv::Mat data(static_cast<int>(good_new.size()), 6, CV_32F);
            for (int i = 0; i < data.rows; ++i) {
                data.at<float>(i, 0) = good_new[i].x;
                data.at<float>(i, 1) = good_new[i].y;
                data.at<float>(i, 2) = uvs[i].x;
                data.at<float>(i, 3) = uvs[i].y;
                data.at<float>(i, 4) = path_vectors[i].x;
                data.at<float>(i, 5) = path_vectors[i].y;
            }

            cv::Mat init_u;
            if (!previous_u_.empty() && previous_n_points_ == data.rows && previous_u_.rows == number_clusters_) {
                init_u = previous_u_;
            }

            if (runFuzzyCMeans(data, number_clusters_, 2.0F, 0.005F, 50, init_u, centers, membership)) {
                previous_u_ = membership.clone();
                previous_n_points_ = data.rows;

                std::vector<int> raw_membership(static_cast<size_t>(data.rows), 0);
                std::vector<float> max_membership(static_cast<size_t>(data.rows), 0.0F);
                for (int j = 0; j < data.rows; ++j) {
                    int best_idx = 0;
                    float best_val = membership.at<float>(0, j);
                    for (int i = 1; i < membership.rows; ++i) {
                        const float v = membership.at<float>(i, j);
                        if (v > best_val) {
                            best_val = v;
                            best_idx = i;
                        }
                    }
                    raw_membership[static_cast<size_t>(j)] = best_idx;
                    max_membership[static_cast<size_t>(j)] = best_val;
                }

                std::vector<bool> outlier_mask(max_membership.size(), false);
                constexpr float membership_threshold = 0.55F;
                for (size_t i = 0; i < max_membership.size(); ++i) {
                    outlier_mask[i] = max_membership[i] < membership_threshold;
                }

                if (!previous_centroids_.empty() && previous_centroids_.rows == number_clusters_) {
                    cv::Mat distances(number_clusters_, previous_centroids_.rows, CV_32F, cv::Scalar(0));
                    for (int i = 0; i < number_clusters_; ++i) {
                        for (int j = 0; j < previous_centroids_.rows; ++j) {
                            const cv::Vec<float, 6> cur(
                                centers.at<float>(i, 0),
                                centers.at<float>(i, 1),
                                centers.at<float>(i, 2),
                                centers.at<float>(i, 3),
                                centers.at<float>(i, 4),
                                centers.at<float>(i, 5)
                            );
                            const cv::Vec<float, 6> prv(
                                previous_centroids_.at<float>(j, 0),
                                previous_centroids_.at<float>(j, 1),
                                previous_centroids_.at<float>(j, 2),
                                previous_centroids_.at<float>(j, 3),
                                previous_centroids_.at<float>(j, 4),
                                previous_centroids_.at<float>(j, 5)
                            );

                            const float pos_dist = std::sqrt((cur[0] - prv[0]) * (cur[0] - prv[0]) + (cur[1] - prv[1]) * (cur[1] - prv[1]));
                            const float vel_dist = std::sqrt((cur[2] - prv[2]) * (cur[2] - prv[2]) + (cur[3] - prv[3]) * (cur[3] - prv[3]));
                            const float path_dist = std::sqrt((cur[4] - prv[4]) * (cur[4] - prv[4]) + (cur[5] - prv[5]) * (cur[5] - prv[5]));
                            distances.at<float>(i, j) = 0.2F * pos_dist + 0.4F * vel_dist + 0.4F * path_dist;
                        }
                    }

                    std::vector<int> new_mapping(static_cast<size_t>(number_clusters_), -1);
                    std::vector<bool> used_prev(static_cast<size_t>(number_clusters_), false);

                    struct PairDist { int curr; int prev; float dist; };
                    std::vector<PairDist> pairs;
                    pairs.reserve(static_cast<size_t>(number_clusters_ * number_clusters_));
                    for (int i = 0; i < number_clusters_; ++i) {
                        for (int j = 0; j < number_clusters_; ++j) {
                            pairs.push_back({i, j, distances.at<float>(i, j)});
                        }
                    }
                    std::sort(pairs.begin(), pairs.end(), [](const PairDist& a, const PairDist& b) {
                        return a.dist < b.dist;
                    });

                    for (const auto& p : pairs) {
                        if (new_mapping[static_cast<size_t>(p.curr)] >= 0 || used_prev[static_cast<size_t>(p.prev)]) {
                            continue;
                        }
                        int mapped = p.prev;
                        if (!cluster_id_mapping_.empty() && p.prev < static_cast<int>(cluster_id_mapping_.size())) {
                            mapped = cluster_id_mapping_[static_cast<size_t>(p.prev)];
                        }
                        new_mapping[static_cast<size_t>(p.curr)] = mapped;
                        used_prev[static_cast<size_t>(p.prev)] = true;
                    }

                    std::vector<bool> used_ids(static_cast<size_t>(number_clusters_), false);
                    for (int mapped : new_mapping) {
                        if (mapped >= 0 && mapped < number_clusters_) {
                            used_ids[static_cast<size_t>(mapped)] = true;
                        }
                    }
                    for (int i = 0; i < number_clusters_; ++i) {
                        if (new_mapping[static_cast<size_t>(i)] >= 0) {
                            continue;
                        }
                        int fallback = i;
                        for (int cand = 0; cand < number_clusters_; ++cand) {
                            if (!used_ids[static_cast<size_t>(cand)]) {
                                fallback = cand;
                                used_ids[static_cast<size_t>(cand)] = true;
                                break;
                            }
                        }
                        new_mapping[static_cast<size_t>(i)] = fallback;
                    }
                    cluster_id_mapping_ = std::move(new_mapping);
                } else {
                    cluster_id_mapping_.assign(static_cast<size_t>(number_clusters_), 0);
                    for (int i = 0; i < number_clusters_; ++i) {
                        cluster_id_mapping_[static_cast<size_t>(i)] = i;
                    }
                }

                for (size_t i = 0; i < labels.size(); ++i) {
                    if (outlier_mask[i]) {
                        labels[i] = -1;
                    } else {
                        const int raw = raw_membership[i];
                        labels[i] = (raw >= 0 && raw < static_cast<int>(cluster_id_mapping_.size())) ? cluster_id_mapping_[static_cast<size_t>(raw)] : raw;
                    }
                }

                previous_centroids_ = centers.clone();
            } else {
                previous_u_.release();
                previous_n_points_ = 0;
            }
        } else {
            previous_u_.release();
            previous_n_points_ = 0;
        }

        cv::Mat mask = cv::Mat::zeros(frame_resized.size(), frame_resized.type());
        cv::Mat result = frame_resized.clone();

        for (size_t i = 0; i < good_new.size(); ++i) {
            const int cluster = (i < labels.size()) ? labels[i] : 0;
            const cv::Scalar color = (cluster == -1)
                ? cv::Scalar(128, 128, 128)
                : colors_[static_cast<size_t>(cluster % static_cast<int>(colors_.size()))];
            const cv::Point pt(static_cast<int>(good_new[i].x), static_cast<int>(good_new[i].y));

            cv::circle(result, pt, 5, color, cv::FILLED);

            if (i < point_ids.size()) {
                point_clusters_[point_ids[i]] = cluster;
            }

            if (i < point_ids.size()) {
                auto it = point_paths_.find(point_ids[i]);
                if (it == point_paths_.end() || it->second.size() <= 1) {
                    continue;
                }
                std::vector<cv::Point> poly;
                poly.reserve(it->second.size());
                for (const auto& p : it->second) {
                    poly.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
                }
                cv::polylines(mask, poly, false, color, 2);
            }
        }

        for (int i = 0; i < centers.rows; ++i) {
            const float cx = centers.at<float>(i, 0);
            const float cy = centers.at<float>(i, 1);
            const float vx = centers.at<float>(i, 2);
            const float vy = centers.at<float>(i, 3);
            const float vel = std::sqrt(vx * vx + vy * vy);

            int mapped_id = i;
            if (!cluster_id_mapping_.empty() && i < static_cast<int>(cluster_id_mapping_.size())) {
                mapped_id = cluster_id_mapping_[static_cast<size_t>(i)];
            }
            const cv::Scalar color = colors_[static_cast<size_t>(mapped_id % static_cast<int>(colors_.size()))];

            std::ostringstream ss;
            ss << "V:" << std::fixed << std::setprecision(1) << vel;
            cv::putText(result, ss.str(), cv::Point(static_cast<int>(cx) + 15, static_cast<int>(cy) - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            cv::putText(result, ss.str(), cv::Point(static_cast<int>(cx) + 15, static_cast<int>(cy) - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

            const cv::Point center_pt(static_cast<int>(cx), static_cast<int>(cy));
            const cv::Point end_pt(static_cast<int>(cx + vx * 0.1F), static_cast<int>(cy + vy * 0.1F));
            cv::arrowedLine(result, center_pt, end_pt, color, 2, cv::LINE_AA, 0, 0.3);
        }

        cv::Mat blended;
        cv::addWeighted(result, 1.0, mask, 0.5, 0.0, blended);
        return blended;
    }

    cv::Mat renderClusteredFlowLegacy(
        const cv::Mat& frame_resized,
        const std::vector<cv::Point2f>& good_new,
        const std::vector<cv::Point2f>& good_old,
        const std::vector<std::deque<cv::Point2f>>& new_paths
    ) {
        std::vector<cv::Point2f> uvs(good_new.size());
        for (size_t i = 0; i < good_new.size(); ++i) {
            uvs[i] = (good_new[i] - good_old[i]) * static_cast<float>(fps_);
        }

        std::vector<cv::Point2f> path_vectors(good_new.size(), cv::Point2f(0.0F, 0.0F));
        for (size_t i = 0; i < new_paths.size(); ++i) {
            if (new_paths[i].empty()) {
                continue;
            }
            cv::Point2f sum(0.0F, 0.0F);
            for (const auto& p : new_paths[i]) {
                sum += p;
            }
            path_vectors[i] = sum * (1.0F / static_cast<float>(new_paths[i].size()));
        }

        std::vector<int> labels(good_new.size(), 0);
        cv::Mat centers;

        if (static_cast<int>(good_new.size()) >= number_clusters_) {
            cv::Mat data(static_cast<int>(good_new.size()), 6, CV_32F);
            for (int i = 0; i < data.rows; ++i) {
                data.at<float>(i, 0) = good_new[i].x;
                data.at<float>(i, 1) = good_new[i].y;
                data.at<float>(i, 2) = uvs[i].x;
                data.at<float>(i, 3) = uvs[i].y;
                data.at<float>(i, 4) = path_vectors[i].x;
                data.at<float>(i, 5) = path_vectors[i].y;
            }

            cv::Mat labels_mat;
            cv::kmeans(
                data,
                number_clusters_,
                labels_mat,
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 0.005),
                3,
                cv::KMEANS_PP_CENTERS,
                centers
            );

            for (int i = 0; i < labels_mat.rows; ++i) {
                labels[i] = labels_mat.at<int>(i, 0);
            }
        }

        cv::Mat mask = cv::Mat::zeros(frame_resized.size(), frame_resized.type());
        cv::Mat result = frame_resized.clone();

        for (size_t i = 0; i < good_new.size(); ++i) {
            const int cluster = (i < labels.size()) ? labels[i] : 0;
            const cv::Scalar color = colors_[cluster % colors_.size()];
            const cv::Point pt(static_cast<int>(good_new[i].x), static_cast<int>(good_new[i].y));

            cv::circle(result, pt, 5, color, cv::FILLED);

            if (i < new_paths.size() && new_paths[i].size() > 1) {
                std::vector<cv::Point> poly;
                poly.reserve(new_paths[i].size());
                for (const auto& p : new_paths[i]) {
                    poly.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
                }
                cv::polylines(mask, poly, false, color, 2);
            }
        }

        for (int i = 0; i < centers.rows; ++i) {
            const float cx = centers.at<float>(i, 0);
            const float cy = centers.at<float>(i, 1);
            const float vx = centers.at<float>(i, 2);
            const float vy = centers.at<float>(i, 3);
            const float vel = std::sqrt(vx * vx + vy * vy);

            std::ostringstream ss;
            ss << "V:" << std::fixed << std::setprecision(1) << vel;
            cv::putText(result, ss.str(), cv::Point(static_cast<int>(cx) + 15, static_cast<int>(cy) - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            cv::putText(result, ss.str(), cv::Point(static_cast<int>(cx) + 15, static_cast<int>(cy) - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        cv::Mat blended;
        cv::addWeighted(result, 1.0, mask, 0.5, 0.0, blended);
        return blended;
    }

#if defined(DEA_ENABLE_VPI) && DEA_ENABLE_VPI
    static void checkVpi(VPIStatus status, const char* where) {
        if (status == VPI_SUCCESS) {
            return;
        }
        char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
        vpiGetLastStatusMessage(buffer, sizeof(buffer));
        throw std::runtime_error(std::string(where) + ": " + vpiStatusGetName(status) + ": " + buffer);
    }

    static void sortVpiKeypoints(VPIArray keypoints, VPIArray scores, int max_points) {
        VPIArrayData pts_data{};
        VPIArrayData scores_data{};
        checkVpi(vpiArrayLockData(keypoints, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &pts_data), "vpiArrayLockData(keypoints)");
        checkVpi(vpiArrayLockData(scores, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &scores_data), "vpiArrayLockData(scores)");

        auto* pts = reinterpret_cast<VPIKeypointF32*>(pts_data.buffer.aos.data);
        auto* scr = reinterpret_cast<uint32_t*>(scores_data.buffer.aos.data);
        const int total = *pts_data.buffer.aos.sizePointer;

        std::vector<int> idx(total);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return scr[a] > scr[b]; });

        const int keep = std::min(total, max_points);
        std::vector<VPIKeypointF32> sorted;
        sorted.reserve(static_cast<size_t>(keep));
        for (int i = 0; i < keep; ++i) {
            sorted.push_back(pts[idx[i]]);
        }
        std::copy(sorted.begin(), sorted.end(), pts);
        *pts_data.buffer.aos.sizePointer = keep;

        checkVpi(vpiArrayUnlock(scores), "vpiArrayUnlock(scores)");
        checkVpi(vpiArrayUnlock(keypoints), "vpiArrayUnlock(keypoints)");
    }

    void initVpiResources(const cv::Mat& first_frame_resized) {
        constexpr uint64_t kVpiMemFlags = static_cast<uint64_t>(VPI_BACKEND_CPU | VPI_BACKEND_CUDA);
        constexpr uint64_t kVpiStreamFlags = static_cast<uint64_t>(VPI_BACKEND_CPU | VPI_BACKEND_CUDA);
        if (vpi_initialized_) {
            checkVpi(vpiImageSetWrappedOpenCVMat(vpi_img_wrap_, first_frame_resized), "vpiImageSetWrappedOpenCVMat");
            return;
        }

        checkVpi(vpiStreamCreate(kVpiStreamFlags, &vpi_stream_), "vpiStreamCreate");
        checkVpi(vpiImageCreateWrapperOpenCVMat(first_frame_resized, kVpiMemFlags, &vpi_img_wrap_), "vpiImageCreateWrapperOpenCVMat");
        checkVpi(vpiImageCreate(first_frame_resized.cols, first_frame_resized.rows, VPI_IMAGE_FORMAT_U8, kVpiMemFlags, &vpi_img_gray_), "vpiImageCreate");

        checkVpi(vpiPyramidCreate(first_frame_resized.cols, first_frame_resized.rows, VPI_IMAGE_FORMAT_U8, vpi_pyramid_levels_, 0.5, kVpiMemFlags, &vpi_pyr_prev_), "vpiPyramidCreate(prev)");
        checkVpi(vpiPyramidCreate(first_frame_resized.cols, first_frame_resized.rows, VPI_IMAGE_FORMAT_U8, vpi_pyramid_levels_, 0.5, kVpiMemFlags, &vpi_pyr_cur_), "vpiPyramidCreate(cur)");

        checkVpi(vpiArrayCreate(max_harris_corners_, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &vpi_prev_features_), "vpiArrayCreate(prev_features)");
        checkVpi(vpiArrayCreate(max_harris_corners_, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &vpi_cur_features_), "vpiArrayCreate(cur_features)");
        checkVpi(vpiArrayCreate(max_harris_corners_, VPI_ARRAY_TYPE_U8, 0, &vpi_status_), "vpiArrayCreate(status)");
        checkVpi(vpiArrayCreate(max_harris_corners_, VPI_ARRAY_TYPE_U32, 0, &vpi_scores_), "vpiArrayCreate(scores)");

        checkVpi(vpiCreateOpticalFlowPyrLK(VPI_BACKEND_CUDA, first_frame_resized.cols, first_frame_resized.rows, VPI_IMAGE_FORMAT_U8, vpi_pyramid_levels_, 0.5, &vpi_optflow_), "vpiCreateOpticalFlowPyrLK");
        checkVpi(vpiInitOpticalFlowPyrLKParams(VPI_BACKEND_CUDA, &vpi_lk_params_), "vpiInitOpticalFlowPyrLKParams");

        VPIStatus harris_status = vpiCreateHarrisCornerDetector(VPI_BACKEND_CUDA, first_frame_resized.cols, first_frame_resized.rows, &vpi_harris_);
        if (harris_status == VPI_SUCCESS) {
            vpi_harris_backend_ = VPI_BACKEND_CUDA;
        } else {
            checkVpi(vpiCreateHarrisCornerDetector(VPI_BACKEND_CPU, first_frame_resized.cols, first_frame_resized.rows, &vpi_harris_), "vpiCreateHarrisCornerDetector(CPU)");
            vpi_harris_backend_ = VPI_BACKEND_CPU;
        }

        vpi_initialized_ = true;
    }

    void initializeTrackingVpi(const cv::Mat& frame_resized) {
        initVpiResources(frame_resized);

        checkVpi(vpiSubmitConvertImageFormat(vpi_stream_, VPI_BACKEND_CUDA, vpi_img_wrap_, vpi_img_gray_, nullptr), "vpiSubmitConvertImageFormat");

        VPIHarrisCornerDetectorParams hparams{};
        checkVpi(vpiInitHarrisCornerDetectorParams(&hparams), "vpiInitHarrisCornerDetectorParams");
        hparams.strengthThresh = 0;
        hparams.sensitivity = 0.01;

        checkVpi(vpiSubmitHarrisCornerDetector(vpi_stream_, vpi_harris_backend_, vpi_harris_, vpi_img_gray_, vpi_cur_features_, vpi_scores_, &hparams), "vpiSubmitHarrisCornerDetector");
        checkVpi(vpiStreamSync(vpi_stream_), "vpiStreamSync(harris)");

        sortVpiKeypoints(vpi_cur_features_, vpi_scores_, max_points_);
        readPointsFromVpiArray(vpi_cur_features_, points_);
        paths_.clear();
        paths_.resize(points_.size());

        checkVpi(vpiSubmitGaussianPyramidGenerator(vpi_stream_, VPI_BACKEND_CUDA, vpi_img_gray_, vpi_pyr_cur_, VPI_BORDER_CLAMP), "vpiSubmitGaussianPyramidGenerator(init)");
        checkVpi(vpiStreamSync(vpi_stream_), "vpiStreamSync(pyramid_init)");
        frame_iter_ = 0;
    }

    void readPointsFromVpiArray(VPIArray arr, std::vector<cv::Point2f>& out_points) {
        VPIArrayData data{};
        checkVpi(vpiArrayLockData(arr, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &data), "vpiArrayLockData(points)");
        const int n = *data.buffer.aos.sizePointer;
        const auto* pts = reinterpret_cast<const VPIKeypointF32*>(data.buffer.aos.data);
        out_points.clear();
        out_points.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            out_points.emplace_back(pts[i].x, pts[i].y);
        }
        checkVpi(vpiArrayUnlock(arr), "vpiArrayUnlock(points)");
    }

    void compactAndExtractVpiTracks(
        std::vector<cv::Point2f>& good_new,
        std::vector<cv::Point2f>& good_old,
        std::vector<std::deque<cv::Point2f>>& new_paths
    ) {
        VPIArrayData cur_data{};
        VPIArrayData prev_data{};
        VPIArrayData status_data{};

        checkVpi(vpiArrayLockData(vpi_cur_features_, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &cur_data), "vpiArrayLockData(cur_features)");
        checkVpi(vpiArrayLockData(vpi_prev_features_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &prev_data), "vpiArrayLockData(prev_features)");
        checkVpi(vpiArrayLockData(vpi_status_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &status_data), "vpiArrayLockData(status)");

        const int total = *cur_data.buffer.aos.sizePointer;
        auto* cur_pts = reinterpret_cast<VPIKeypointF32*>(cur_data.buffer.aos.data);
        const auto* prev_pts = reinterpret_cast<const VPIKeypointF32*>(prev_data.buffer.aos.data);
        const auto* st = reinterpret_cast<const uint8_t*>(status_data.buffer.aos.data);

        good_new.clear();
        good_old.clear();
        new_paths.clear();
        good_new.reserve(static_cast<size_t>(total));
        good_old.reserve(static_cast<size_t>(total));
        new_paths.reserve(static_cast<size_t>(total));

        int w = 0;
        for (int i = 0; i < total; ++i) {
            // VPI status: 0 means tracking succeeded.
            if (st[i] != 0) {
                continue;
            }
            const cv::Point2f cur_p(cur_pts[i].x, cur_pts[i].y);
            const cv::Point2f prev_p(prev_pts[i].x, prev_pts[i].y);

            good_new.push_back(cur_p);
            good_old.push_back(prev_p);

            std::deque<cv::Point2f> path;
            if (i < static_cast<int>(paths_.size())) {
                path = paths_[i];
            }
            path.push_back(cur_p);
            while (static_cast<int>(path.size()) > max_path_length_) {
                path.pop_front();
            }
            new_paths.push_back(std::move(path));

            cur_pts[w++] = cur_pts[i];
        }
        *cur_data.buffer.aos.sizePointer = w;

        checkVpi(vpiArrayUnlock(vpi_status_), "vpiArrayUnlock(status)");
        checkVpi(vpiArrayUnlock(vpi_prev_features_), "vpiArrayUnlock(prev_features)");
        checkVpi(vpiArrayUnlock(vpi_cur_features_), "vpiArrayUnlock(cur_features)");
    }

    cv::Mat processFrameVpi(const cv::Mat& frame_resized) {
        if (!vpi_initialized_ || points_.empty()) {
            initializeTrackingVpi(frame_resized);
            return frame_resized;
        }

        checkVpi(vpiImageSetWrappedOpenCVMat(vpi_img_wrap_, frame_resized), "vpiImageSetWrappedOpenCVMat(frame)");

        std::swap(vpi_prev_features_, vpi_cur_features_);
        std::swap(vpi_pyr_prev_, vpi_pyr_cur_);

        checkVpi(vpiSubmitConvertImageFormat(vpi_stream_, VPI_BACKEND_CUDA, vpi_img_wrap_, vpi_img_gray_, nullptr), "vpiSubmitConvertImageFormat(frame)");
        checkVpi(vpiSubmitGaussianPyramidGenerator(vpi_stream_, VPI_BACKEND_CUDA, vpi_img_gray_, vpi_pyr_cur_, VPI_BORDER_CLAMP), "vpiSubmitGaussianPyramidGenerator(frame)");
        checkVpi(vpiSubmitOpticalFlowPyrLK(vpi_stream_, 0, vpi_optflow_, vpi_pyr_prev_, vpi_pyr_cur_, vpi_prev_features_, vpi_cur_features_, vpi_status_, &vpi_lk_params_), "vpiSubmitOpticalFlowPyrLK");
        checkVpi(vpiStreamSync(vpi_stream_), "vpiStreamSync(flow)");

        std::vector<cv::Point2f> good_new;
        std::vector<cv::Point2f> good_old;
        std::vector<std::deque<cv::Point2f>> new_paths;
        compactAndExtractVpiTracks(good_new, good_old, new_paths);

        if (good_new.empty()) {
            initializeTrackingVpi(frame_resized);
            return frame_resized;
        }

        cv::Mat blended = renderClusteredFlowLegacy(frame_resized, good_new, good_old, new_paths);
        points_ = good_new;
        paths_ = std::move(new_paths);
        frame_iter_ += 1;

        if (static_cast<int>(points_.size()) < std::max(8, number_clusters_)) {
            initializeTrackingVpi(frame_resized);
        }

        return blended;
    }

    void releaseVpi() {
        if (vpi_stream_ != nullptr) {
            vpiStreamDestroy(vpi_stream_);
            vpi_stream_ = nullptr;
        }
        if (vpi_harris_ != nullptr) {
            vpiPayloadDestroy(vpi_harris_);
            vpi_harris_ = nullptr;
        }
        if (vpi_optflow_ != nullptr) {
            vpiPayloadDestroy(vpi_optflow_);
            vpi_optflow_ = nullptr;
        }
        if (vpi_pyr_prev_ != nullptr) {
            vpiPyramidDestroy(vpi_pyr_prev_);
            vpi_pyr_prev_ = nullptr;
        }
        if (vpi_pyr_cur_ != nullptr) {
            vpiPyramidDestroy(vpi_pyr_cur_);
            vpi_pyr_cur_ = nullptr;
        }
        if (vpi_img_wrap_ != nullptr) {
            vpiImageDestroy(vpi_img_wrap_);
            vpi_img_wrap_ = nullptr;
        }
        if (vpi_img_gray_ != nullptr) {
            vpiImageDestroy(vpi_img_gray_);
            vpi_img_gray_ = nullptr;
        }
        if (vpi_prev_features_ != nullptr) {
            vpiArrayDestroy(vpi_prev_features_);
            vpi_prev_features_ = nullptr;
        }
        if (vpi_cur_features_ != nullptr) {
            vpiArrayDestroy(vpi_cur_features_);
            vpi_cur_features_ = nullptr;
        }
        if (vpi_status_ != nullptr) {
            vpiArrayDestroy(vpi_status_);
            vpi_status_ = nullptr;
        }
        if (vpi_scores_ != nullptr) {
            vpiArrayDestroy(vpi_scores_);
            vpi_scores_ = nullptr;
        }
        vpi_initialized_ = false;
    }
#endif

    int number_clusters_{3};
    double fps_{30.0};
    cv::Size processing_size_{640, 480};
    bool flow_gpu_requested_{false};
    bool use_vpi_gpu_{false};

    int max_points_{30};
    int max_path_length_{20};
    cv::TermCriteria lk_criteria_{cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.03};

    cv::Mat old_gray_;
    int next_point_id_{0};
    std::unordered_map<int, cv::Point2f> tracked_points_;
    std::unordered_map<int, std::deque<cv::Point2f>> point_paths_;
    std::unordered_map<int, int> point_clusters_;
    cv::Mat detection_mask_;
    int mask_recovery_rate_{2};
    int invalid_region_radius_{20};
    cv::Mat previous_centroids_;
    std::vector<int> cluster_id_mapping_;
    cv::Mat previous_u_;
    int previous_n_points_{0};

    std::vector<cv::Point2f> points_;
    std::vector<std::deque<cv::Point2f>> paths_;
    int frame_iter_{0};

    std::vector<cv::Scalar> colors_;

#if defined(DEA_ENABLE_VPI) && DEA_ENABLE_VPI
    bool vpi_initialized_{false};
    int max_harris_corners_{2048};
    int vpi_pyramid_levels_{4};
    VPIBackend vpi_harris_backend_{VPI_BACKEND_CPU};
    VPIStream vpi_stream_{nullptr};
    VPIImage vpi_img_wrap_{nullptr};
    VPIImage vpi_img_gray_{nullptr};
    VPIPyramid vpi_pyr_prev_{nullptr};
    VPIPyramid vpi_pyr_cur_{nullptr};
    VPIArray vpi_prev_features_{nullptr};
    VPIArray vpi_cur_features_{nullptr};
    VPIArray vpi_status_{nullptr};
    VPIArray vpi_scores_{nullptr};
    VPIPayload vpi_harris_{nullptr};
    VPIPayload vpi_optflow_{nullptr};
    VPIOpticalFlowPyrLKParams vpi_lk_params_{};
#endif
};

}  // namespace

int main(int argc, char** argv) {
    configureRuntime();

    Args args;
    bool show_help = false;
    if (!parseArgs(argc, argv, args, show_help)) {
        printUsage(argv[0]);
        return 1;
    }
    if (show_help) {
        printUsage(argv[0]);
        return 0;
    }

    if (args.no_display && args.output.empty()) {
        std::cout << "Warning: --no-display sem --output roda apenas como benchmark." << '\n';
    }

    std::cout << "=== DeACpp Optimized Pipeline ===" << '\n';

    std::string rtsp_url;
    if (!args.video_file.empty()) {
        std::cout << "Input file: " << args.video_file << '\n';
    } else {
        rtsp_url = buildRtspUrl(args.video_ip, args.video_port, args.video_path);
        std::cout << "RTSP: " << rtsp_url << '\n';
        std::cout << "RTSP opts: backend=" << args.rtsp_backend
                  << " transport=" << args.rtsp_transport
                  << " latency_ms=" << args.rtsp_latency_ms
                  << " open_timeout_ms=" << args.rtsp_open_timeout_ms
                  << " first_frame_timeout=" << args.rtsp_first_frame_timeout
                  << " max_timeouts=" << args.rtsp_max_consecutive_timeouts << '\n';
    }

    cv::VideoCapture cap;
    cv::VideoWriter writer;

    std::unique_ptr<LatestFrameReader> frame_reader;
    std::unique_ptr<ModuleWorker> yolo_worker;
    std::unique_ptr<ModuleWorker> sky_worker;
    std::unique_ptr<ModuleWorker> flow_worker;

    try {
        std::string backend_used;
        std::string open_error;
        double raw_fps = 0.0;
        int src_w = 0;
        int src_h = 0;

        bool opened = false;
        if (!args.video_file.empty()) {
            opened = openFileCapture(args.video_file, cap, backend_used, raw_fps, src_w, src_h, open_error);
        } else {
            opened = openCapture(
                rtsp_url,
                args.rtsp_backend,
                args.rtsp_transport,
                args.rtsp_latency_ms,
                args.rtsp_open_timeout_ms,
                cap,
                backend_used,
                raw_fps,
                src_w,
                src_h,
                open_error
            );
        }
        if (!opened) {
            throw std::runtime_error("Could not open capture: " + open_error);
        }

        std::cout << "Capture backend: " << backend_used << '\n';
        std::cout << "Source size: " << src_w << "x" << src_h << " | Reported FPS: " << raw_fps << '\n';

        if (src_h <= 0 || src_w <= 0) {
            throw std::runtime_error("Source size is invalid after capture probe.");
        }

        const int proc_h = std::max(1, args.resize_height);
        const int proc_w = static_cast<int>(src_w * (proc_h / static_cast<double>(src_h)));
        if (proc_w <= 0) {
            throw std::runtime_error("Invalid processing width computed.");
        }
        const cv::Size proc_size(proc_w, proc_h);

        const double algo_fps = resolveFps(raw_fps, 30.0);
        const double output_fps = std::max(1.0, args.output_fps);

        std::cout << "Processing size: " << proc_w << "x" << proc_h << '\n';
        std::cout << "Algorithm FPS reference: " << algo_fps << '\n';
        std::cout << "Output FPS: " << output_fps << '\n';

        if (!args.output.empty()) {
            const fs::path out_path(args.output);
            if (out_path.has_parent_path()) {
                fs::create_directories(out_path.parent_path());
            }
            const int view_count = 1 + (args.disable_sky ? 0 : 1) + (args.disable_flow ? 0 : 1);
            const int out_w = proc_w * std::max(1, view_count);
            const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer.open(args.output, fourcc, output_fps, cv::Size(out_w, proc_h));
            if (!writer.isOpened()) {
                throw std::runtime_error("Failed to open VideoWriter: " + args.output);
            }
        }

        const bool render_enabled = !args.no_display || writer.isOpened();
        const bool use_latest_frame_policy = args.video_file.empty();
        if (use_latest_frame_policy) {
            frame_reader = std::make_unique<LatestFrameReader>(&cap);
            frame_reader->start();
            std::cout << "Frame policy: latest-frame (drop enabled for live source)." << '\n';
        } else {
            std::cout << "Frame policy: sequential (no drop for video file)." << '\n';
        }

        std::cout << "\n--- Initializing modules ---\n";

        auto yolo_detector = std::make_shared<YOLODetector>(
            args.yolo_model_path,
            args.confidence,
            50,
            1.1,
            1.5,
            1.5,
            "# ALERTA: APROXIMACAO DETECTADA"
        );

        std::shared_ptr<SkySegmentation> sky_seg;
        if (!args.disable_sky) {
            sky_seg = std::make_shared<SkySegmentation>(
                args.sky_model_path,
                cv::Size(320, 320),
                1,
                30,
                0.75,
                0.25,
                128
            );
        }

        std::shared_ptr<OpticalFlowModule> flow_mod;
        if (!args.disable_flow) {
            flow_mod = std::make_shared<OpticalFlowModule>(args.clusters, algo_fps, proc_size, args.flow_gpu);
        }

        std::cout << "Modules initialized." << '\n';

        yolo_worker = std::make_unique<ModuleWorker>("yolo", [yolo_detector](const cv::Mat& f) {
            return yolo_detector->processFrame(f);
        });
        yolo_worker->start();

        if (sky_seg) {
            sky_worker = std::make_unique<ModuleWorker>("sky", [sky_seg](const cv::Mat& f) {
                return sky_seg->processFrame(f);
            });
            sky_worker->start();
        }

        if (flow_mod) {
            flow_worker = std::make_unique<ModuleWorker>("flow", [flow_mod](const cv::Mat& f) {
                return flow_mod->processFrame(f);
            });
            flow_worker->start();
        }

        std::cout << "\n--- Running ---\n";
        std::cout << "Press 'q' or ESC to exit." << '\n';

        int64_t frame_count = 0;
        int64_t last_frame_id = 0;
        const double start_ts = nowSeconds();
        double last_stats_ts = start_ts;
        bool summary_printed = false;
        int consecutive_read_timeouts = 0;
        const double first_frame_deadline = start_ts + std::max(1.0, args.rtsp_first_frame_timeout);
        const int max_consecutive_timeouts = std::max(1, args.rtsp_max_consecutive_timeouts);

        double writer_next_ts = nowSeconds();

        while (true) {
            cv::Mat frame;
            int64_t frame_id = 0;
            double frame_ts = 0.0;
            bool ok = false;
            if (use_latest_frame_policy) {
                ok = frame_reader->getLatest(last_frame_id, args.read_timeout, frame, frame_id, frame_ts);
            } else {
                ok = cap.read(frame);
                if (ok && !frame.empty()) {
                    frame_id = last_frame_id + 1;
                    frame_ts = nowSeconds();
                } else {
                    frame.release();
                    ok = false;
                }
            }
            if (!ok) {
                if (use_latest_frame_policy) {
                    consecutive_read_timeouts += 1;
                    const double now = nowSeconds();
                    if (frame_count == 0 && now < first_frame_deadline) {
                        if (consecutive_read_timeouts == 1 || (consecutive_read_timeouts % 3) == 0) {
                            const double remain = std::max(0.0, first_frame_deadline - now);
                            std::cout << std::fixed << std::setprecision(1)
                                      << "Waiting first RTSP frame... remaining " << remain << "s\n";
                        }
                        continue;
                    }
                    if (consecutive_read_timeouts < max_consecutive_timeouts) {
                        std::cout << "No new RTSP frame in timeout window (" << consecutive_read_timeouts
                                  << "/" << max_consecutive_timeouts << "). Retrying..." << '\n';
                        continue;
                    }
                    std::cout << "No new frames received in time. Stopping processing loop." << '\n';
                } else {
                    std::cout << "End of video file reached. Stopping processing loop." << '\n';
                }
                const double total_time = nowSeconds() - start_ts;
                const double avg_fps = total_time > 0.0 ? static_cast<double>(frame_count) / total_time : 0.0;
                const int64_t total_read = use_latest_frame_policy
                    ? (frame_reader ? frame_reader->totalRead() : 0)
                    : frame_count;
                const int64_t dropped = std::max<int64_t>(0, total_read - frame_count);
                std::cout << "--- Processing completed ---" << '\n';
                std::cout << "Total frames processed: " << frame_count << '\n';
                std::cout << "Total frames read: " << total_read << " (dropped: " << dropped << ")\n";
                std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << "s\n";
                std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << avg_fps << '\n';
                summary_printed = true;
                break;
            }

            consecutive_read_timeouts = 0;
            last_frame_id = frame_id;
            frame_count += 1;

            cv::Mat resized;
            cv::resize(frame, resized, proc_size, 0, 0, cv::INTER_AREA);

            if (frame_count % std::max<int64_t>(1, args.yolo_update_interval) == 0) {
                yolo_worker->submit(resized, frame_id);
            }
            if (sky_worker && frame_count % std::max<int64_t>(1, args.sky_update_interval) == 0) {
                sky_worker->submit(resized, frame_id);
            }
            if (flow_worker && frame_count % std::max<int64_t>(1, args.flow_update_interval) == 0) {
                flow_worker->submit(resized, frame_id);
            }

            WorkerSnapshot yolo_snap;
            WorkerSnapshot sky_snap;
            WorkerSnapshot flow_snap;
            if (render_enabled) {
                yolo_snap = yolo_worker->getLatestOutput();
                if (sky_worker) {
                    sky_snap = sky_worker->getLatestOutput();
                }
                if (flow_worker) {
                    flow_snap = flow_worker->getLatestOutput();
                }
            } else {
                const WorkerMetrics yolo_meta = yolo_worker->getLatestMetrics();
                yolo_snap.frame_id = yolo_meta.frame_id;
                yolo_snap.proc_ms = yolo_meta.proc_ms;
                yolo_snap.error = yolo_meta.error;
                yolo_snap.total_processed = yolo_meta.total_processed;

                if (sky_worker) {
                    const WorkerMetrics sky_meta = sky_worker->getLatestMetrics();
                    sky_snap.frame_id = sky_meta.frame_id;
                    sky_snap.proc_ms = sky_meta.proc_ms;
                    sky_snap.error = sky_meta.error;
                    sky_snap.total_processed = sky_meta.total_processed;
                }
                if (flow_worker) {
                    const WorkerMetrics flow_meta = flow_worker->getLatestMetrics();
                    flow_snap.frame_id = flow_meta.frame_id;
                    flow_snap.proc_ms = flow_meta.proc_ms;
                    flow_snap.error = flow_meta.error;
                    flow_snap.total_processed = flow_meta.total_processed;
                }
            }

            const double now = nowSeconds();
            const double elapsed = std::max(1e-6, now - start_ts);
            const double loop_fps = static_cast<double>(frame_count) / elapsed;
            const double cap_fps = use_latest_frame_policy
                ? static_cast<double>(frame_reader->totalRead()) / elapsed
                : loop_fps;
            const double latency_ms = std::max(0.0, (now - frame_ts) * 1000.0);

            if (render_enabled) {
                cv::Mat yolo_view = ensure3ch(yolo_snap.frame, resized);
                std::vector<cv::Mat> views;
                views.reserve(3);
                views.push_back(yolo_view);
                if (sky_worker) {
                    views.push_back(ensure3ch(sky_snap.frame, resized));
                }
                if (flow_worker) {
                    views.push_back(ensure3ch(flow_snap.frame, resized));
                }

                cv::Mat combined;
                if (views.size() == 1) {
                    combined = views[0];
                } else {
                    cv::hconcat(views, combined);
                }

                {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(1)
                       << "in:" << cap_fps << "fps loop:" << loop_fps
                       << " latency:" << std::setprecision(0) << latency_ms << "ms"
                       << " frame:" << frame_count;
                    putShadowText(combined, ss.str(), cv::Point(10, 26), 0.6);
                }
                {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(1)
                       << "YOLO " << yolo_snap.proc_ms << "ms lag:" << std::max<int64_t>(0, frame_id - yolo_snap.frame_id)
                       << " proc:" << yolo_snap.total_processed;
                    putShadowText(combined, ss.str(), cv::Point(10, 52), 0.55);
                }
                {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(1)
                       << "SKY  " << sky_snap.proc_ms << "ms lag:" << std::max<int64_t>(0, frame_id - sky_snap.frame_id)
                       << " proc:" << sky_snap.total_processed;
                    putShadowText(combined, ss.str(), cv::Point(10, 76), 0.55);
                }
                {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(1)
                       << "FLOW " << flow_snap.proc_ms << "ms lag:" << std::max<int64_t>(0, frame_id - flow_snap.frame_id)
                       << " proc:" << flow_snap.total_processed;
                    putShadowText(combined, ss.str(), cv::Point(10, 100), 0.55);
                }

                if (!yolo_snap.error.empty()) {
                    putShadowText(
                        combined,
                        "YOLO err: " + truncateText(yolo_snap.error, 90),
                        cv::Point(10, proc_h - 54),
                        0.5,
                        cv::Scalar(0, 0, 255)
                    );
                }
                if (!sky_snap.error.empty()) {
                    putShadowText(
                        combined,
                        "SKY err: " + truncateText(sky_snap.error, 90),
                        cv::Point(10, proc_h - 32),
                        0.5,
                        cv::Scalar(0, 0, 255)
                    );
                }
                if (!flow_snap.error.empty()) {
                    putShadowText(
                        combined,
                        "FLOW err: " + truncateText(flow_snap.error, 90),
                        cv::Point(10, proc_h - 10),
                        0.5,
                        cv::Scalar(0, 0, 255)
                    );
                }

                if (writer.isOpened()) {
                    if ((now - writer_next_ts) > 1.0) {
                        writer_next_ts = now;
                    }
                    while (now >= writer_next_ts) {
                        writer.write(combined);
                        writer_next_ts += 1.0 / output_fps;
                    }
                }

                if (!args.no_display) {
                    cv::imshow("DeACpp - YOLO | SkySeg | OpticalFlow", combined);
                    const int key = cv::waitKey(1) & 0xFF;
                    if (key == 27 || key == 'q') {
                        break;
                    }
                }
            }

            if ((now - last_stats_ts) >= args.stats_interval) {
                std::cout << std::fixed << std::setprecision(1)
                          << "[stats] in=" << cap_fps << "fps loop=" << loop_fps
                          << " yolo=" << yolo_snap.proc_ms << "ms"
                          << " sky=" << sky_snap.proc_ms << "ms"
                          << " flow=" << flow_snap.proc_ms << "ms"
                          << " lag(y/s/f)=(" << std::max<int64_t>(0, frame_id - yolo_snap.frame_id)
                          << "/" << std::max<int64_t>(0, frame_id - sky_snap.frame_id)
                          << "/" << std::max<int64_t>(0, frame_id - flow_snap.frame_id)
                          << ")\n";
                last_stats_ts = now;
            }
        }

        if (!summary_printed) {
            const double total_time = nowSeconds() - start_ts;
            const double avg_fps = total_time > 0.0 ? static_cast<double>(frame_count) / total_time : 0.0;
            const int64_t total_read = use_latest_frame_policy
                ? (frame_reader ? frame_reader->totalRead() : 0)
                : frame_count;
            const int64_t dropped = std::max<int64_t>(0, total_read - frame_count);
            std::cout << "--- Processing completed ---" << '\n';
            std::cout << "Total frames processed: " << frame_count << '\n';
            std::cout << "Total frames read: " << total_read << " (dropped: " << dropped << ")\n";
            std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << "s\n";
            std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << avg_fps << '\n';
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';

        if (frame_reader) {
            frame_reader->stop();
        }
        if (yolo_worker) {
            yolo_worker->stop();
        }
        if (sky_worker) {
            sky_worker->stop();
        }
        if (flow_worker) {
            flow_worker->stop();
        }

        if (cap.isOpened()) {
            cap.release();
        }
        if (writer.isOpened()) {
            writer.release();
        }
        cv::destroyAllWindows();
        return 1;
    }

    if (frame_reader) {
        frame_reader->stop();
    }
    if (yolo_worker) {
        yolo_worker->stop();
    }
    if (sky_worker) {
        sky_worker->stop();
    }
    if (flow_worker) {
        flow_worker->stop();
    }

    if (cap.isOpened()) {
        cap.release();
    }
    if (writer.isOpened()) {
        writer.release();
    }

    cv::destroyAllWindows();
    return 0;
}
