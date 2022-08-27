#ifndef YOLACT_UINT8_H_
#define YOLACT_UINT8_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "tengine_operations.h"

#ifdef __cplusplus
extern "C" {
#endif

enum Status {
	waiting, inferencing, inferenced
};

struct Frames {
	cv::Mat img;
	volatile Status status = waiting;
	unsigned long long frame_num = UINT64_MAX;
};

class Model {
    Frames* frames;
    void* graph;
    void* input_tensor;
public:
    float thresh = 0.3; 
    Model(const char* model_file, Frames* frames_);
    int inference();
    void operator()() {
        while (true) {
            if (frames->status == inferencing)
                inference();   
        }
    }
};

#ifdef __cplusplus
}
#endif

#endif /* YOLACT_UINT8_H_ */