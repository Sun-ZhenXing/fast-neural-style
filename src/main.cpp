#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

const float VGG_MEAN_B = 103.939f;
const float VGG_MEAN_G = 116.779f;
const float VGG_MEAN_R = 123.680f;

cv::dnn::Net load_model(const std::string& model_path) {
    cv::dnn::Net net = cv::dnn::readNetFromTorch(model_path);
    if (net.empty()) {
        cout << "[Error] load model failed!" << endl;
        exit(-1);
    }
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cout << ">> use cuda" << endl;
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    } else {
        cout << ">> use cpu" << endl;
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    }
    return net;
}

cv::Mat process(cv::Mat& image, cv::dnn::Net& net) {
    int h = image.rows;
    int w = image.cols;
    cv::Mat blob = cv::dnn::blobFromImage(
        image, 1.0, cv::Size(w, h),
        cv::Scalar(VGG_MEAN_B, VGG_MEAN_G, VGG_MEAN_R), false, false);
    net.setInput(blob);
    cv::Mat out = net.forward();
    auto p_image = out.ptr<float>();
    int image_size = out.size[2] * out.size[3];

    cv::Mat b(out.size[2], out.size[3], CV_32FC1, p_image);
    b += VGG_MEAN_B;
    cv::Mat g(out.size[2], out.size[3], CV_32FC1, p_image + image_size);
    g += VGG_MEAN_G;
    cv::Mat r(out.size[2], out.size[3], CV_32FC1, p_image + image_size * 2);
    r += VGG_MEAN_R;

    vector<cv::Mat> channels = {b, g, r};
    cv::Mat res;
    cv::merge(channels, res);
    res.convertTo(res, CV_8UC3);
    return res;
}

int show_frame(cv::Mat& frame, cv::Mat& out) {
    cv::Mat merge(frame.rows, frame.cols * 2, CV_8UC3);
    cv::Mat left(merge, cv::Rect(0, 0, frame.cols, frame.rows));
    frame.copyTo(left);
    cv::Mat right(merge, cv::Rect(frame.cols, 0, frame.cols, frame.rows));
    out.copyTo(right);
    cv::imshow("demo", merge);
    int key = cv::waitKey(1) & 0xff;
    if (key == 's') {
        cv::imwrite("./out.jpg", merge);
    }
    return key;
}

int main(int argc, char* argv[]) {
    cout << ">> press q to exit." << endl;
    cout << ">> press s to save frame." << endl;
    if (argc < 2) {
        cout << "Usage: $ " << argv[0] << " <model_path>" << endl;
        return -1;
    }
    auto net = load_model(argv[1]);
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "[Error] open camera failed!" << endl;
        return -1;
    }
    while (cap.isOpened()) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cout << "[Error] frame is empty!" << endl;
            break;
        }
        cv::Mat out = process(frame, net);
        if (show_frame(frame, out) == 'q') break;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
