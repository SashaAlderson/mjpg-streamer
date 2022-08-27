#include <vector>
#include <iostream>
#include <future>

#include "yolact_uint8.h"
#include "tengine/c_api.h"
#include "common.h"

#define TARGET_SIZE 544
#define VXDEVICE  "TIMVX"
#define VERBOSE false
#define DEBUG 0
#define CORRECT_MASK 1 // move mask to right bottom corner
#define OFFSET 6 // offset for mask

const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
const float norm_vals[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> maskdata;
    cv::Mat mask;
};

void get_input_data_cv_uint8(const cv::Mat& sample, uint8_t* input_data, int img_h, int img_w, const float* mean, const float* scale, 
                       float input_scale, int zero_point) {
    cv::Mat img;
    if (sample.channels() == 4) {
        cv::cvtColor(sample, img, cv::COLOR_BGRA2RGB);
    }
    else if (sample.channels() == 1) {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    }
    else if (sample.channels() == 3) {
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
    }
    else {
        img = sample;
    }

    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float* img_data = (float* )img.data;

    /* nhwc to nchw */
    for (int h = 0; h < img_h; h++) {  
        for (int w = 0; w < img_w; w++) {
            for (int c = 0; c < 3; c++) {
                int in_index  = h * img_w * 3 + w * 3 + c;
                int out_index = c * img_h * img_w + h * img_w + w;
                float input_fp32 = (img_data[in_index] - mean[c]) * scale[c];
                /* quant to uint8 */
                int udata = (round)(input_fp32 / input_scale + ( float )zero_point);
                if (udata > 255)
                    udata = 255;
                else if (udata < 0)
                    udata = 0;

                input_data[out_index] = udata;
            }
        }
    }
}

struct Box2f {
    float cx;
    float cy;
    float w;
    float h;
};

static std::vector<Box2f> generate_priorbox(int num_priores) {

    std::vector<Box2f> priorboxs(num_priores);

    const int conv_ws[5] = {68, 34, 17, 9, 5};
    const int conv_hs[5] = {68, 34, 17, 9, 5};

    const float aspect_ratios[3] = {1.0f, 0.5f, 2.f};
    const float scales[5] = {24.f, 48.f, 96.f, 192.f, 384.f};

    int index = 0;

    for (int i = 0; i < 5; i++) {
        int conv_w = conv_ws[i];
        int conv_h = conv_hs[i];
        int scale = scales[i];
        for (int ii = 0; ii < conv_h; ii++) {
            for (int j = 0; j < conv_w; j++) {
                float cx = (j + 0.5f) / conv_w;
                float cy = (ii + 0.5f) / conv_h;

                for (int k = 0; k < 3; k++) {
                    float ar = aspect_ratios[k];

                    ar = sqrt(ar);

                    float w = scale * ar / TARGET_SIZE;
                    float h = scale / ar / TARGET_SIZE;

                    // h = w;

                    Box2f& priorbox = priorboxs[index];

                    priorbox.cx = cx;
                    priorbox.cy = cy;
                    priorbox.w = w;
                    priorbox.h = h;

                    index += 1;
                }
            }
        }
    }

    return priorboxs;
}

static inline float intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void fast_nms(std::vector<std::vector<Object>>& class_candidates, std::vector<Object>& objects,
                     const float iou_thresh, const int nms_top_k, const int keep_top_k) {
    for (int i = 0; i < ( int )class_candidates.size(); i++)
    {
        std::vector<Object>& candidate = class_candidates[i];
        std::sort(candidate.begin(), candidate.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
        if (candidate.size() == 0)
            continue;

        if (nms_top_k != 0 && nms_top_k > candidate.size()) {
            candidate.erase(candidate.begin() + nms_top_k, candidate.end());
        }

        objects.push_back(candidate[0]);
        const int n = candidate.size();
        std::vector<float> areas(n);
        std::vector<int> keep(n);
        for (int j = 0; j < n; j++) {
            areas[j] = candidate[j].rect.area();
        }
        std::vector<std::vector<float>> iou_matrix;
        for (int j = 0; j < n; j++) {
            std::vector<float> iou_row(n);
            for (int k = 0; k < n; k++) {
                float inter_area = intersection_area(candidate[j], candidate[k]);
                float union_area = areas[j] + areas[k] - inter_area;
                iou_row[k] = inter_area / union_area;
            }
            iou_matrix.push_back(iou_row);
        }
        for (int j = 1; j < n; j++) {
            std::vector<float>::iterator max_value;
            max_value = std::max_element(iou_matrix[j].begin(), iou_matrix[j].begin() + j - 1);
            if (*max_value <= iou_thresh) {
                objects.push_back(candidate[j]);
            }
        }
    }
    std::sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
    if (objects.size() > keep_top_k)
        objects.resize(keep_top_k);
}

static void softmax(std::vector<float> &tensor) {
    double sum = 0;
    double maxcls = 0.1;
    // int maxind = 0;
    for (std::size_t i = 0; i < tensor.size()/81; i++) {
        sum = 0;
        maxcls = 0.1;
        for (int j = 0; j < 81; j++) {
            sum += tensor[i*81 + j]; // moved exp to another place
        }
        for (int j = 0; j < 81; j++) {
            tensor[i*81 + j] = tensor[i*81 + j] / sum; // moved exp to another place
            if (tensor[i*81 + j] > maxcls) {
                // maxind = j;
                maxcls = tensor[i*81 + j];
            }
        }
    }
}

int set_graph(int img_h, int img_w, graph_t graph){
        /* set the input shape to initial the graph, and prerun graph to infer shape */
    
    int dims[] = {1, 3, img_h, img_w};    // nchw
     

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr) {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0) {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }



    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph(graph) < 0) {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }
    return 0;
}

int detect_yolact(const cv::Mat& bgr, std::vector<Object>& objects, graph_t graph, tensor_t input_tensor, float confidence_thresh) {   

    // double start = get_current_time();
    
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // /* set the input shape to initial the graph, and prerun graph to infer shape */
    const int target_size = TARGET_SIZE;
    int img_size = target_size * target_size * 3;
    // int dims[] = {1, 3, target_size, target_size};    // nchw
    std::vector<uint8_t> input_data(img_size);

    if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0) {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }    
    // if (VERBOSE) {
    //     std::cout << "Preparations:" << get_current_time() - start << std::endl;
    // }

    /* prepare process input data, set the data mem to input tensor */
    float input_scale = 0.f;
    int input_zero_point = 0;
    get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);   
    get_input_data_cv_uint8(bgr, input_data.data(), target_size, target_size, mean_vals, norm_vals, input_scale, input_zero_point);
    
    if (run_graph(graph, 1) < 0) {
        fprintf(stderr, "Run graph failed\n");
        return -1;
    }
    // if (VERBOSE) {
    //     std::cout << "Inference:" << get_current_time() - start << std::endl;
    // }
    if (DEBUG) {
        int out_dim[4];
        for (int i = 0; i < 4; i++) {
            get_tensor_shape( get_graph_output_tensor(graph, i, 0), out_dim, 4);
            std::cout << "Shape of " << i << "'th tensor: " << out_dim[0] << " "<< out_dim[1] << " " << out_dim[2] << " "<< out_dim[3] << std::endl;
        }
    }

    tensor_t maskmaps_tensor   = get_graph_output_tensor(graph, 0, 0);
    tensor_t location_tensor   = get_graph_output_tensor(graph, 2, 0); // 2
    tensor_t mask_tensor       = get_graph_output_tensor(graph, 3, 0); // 3
    tensor_t confidence_tensor = get_graph_output_tensor(graph, 1, 0); // 1

    float maskmaps_scale = 0.f;
    float location_scale = 0.f;
    float mask_scale     = 0.f;
    float confidence_scale = 0.f;

    int maskmaps_zero_point = 0;
    int location_zero_point = 0;
    int mask_zero_point     = 0;
    int confidence_zero_point = 0;

    get_tensor_quant_param(maskmaps_tensor, &maskmaps_scale, &maskmaps_zero_point, 1);
    get_tensor_quant_param(location_tensor, &location_scale, &location_zero_point, 1);
    get_tensor_quant_param(mask_tensor, &mask_scale, &mask_zero_point, 1);
    get_tensor_quant_param(confidence_tensor, &confidence_scale, &confidence_zero_point, 1);

    int maskmaps_count   = get_tensor_buffer_size(maskmaps_tensor) / sizeof(uint8_t);
    int location_count   = get_tensor_buffer_size(location_tensor) / sizeof(uint8_t);
    int mask_count       = get_tensor_buffer_size(mask_tensor) / sizeof(uint8_t);
    int confidence_count = get_tensor_buffer_size(confidence_tensor) / sizeof(uint8_t);

    uint8_t* maskmaps_u8   = ( uint8_t* )get_tensor_buffer(maskmaps_tensor);
    uint8_t* location_u8   = ( uint8_t* )get_tensor_buffer(location_tensor);
    uint8_t* mask_u8       = ( uint8_t* )get_tensor_buffer(mask_tensor);
    uint8_t* confidence_u8 = ( uint8_t* )get_tensor_buffer(confidence_tensor);

    std::vector<float> maskmaps(maskmaps_count);
    std::vector<float> location(location_count);
    std::vector<float> mask(mask_count);
    std::vector<float> confidence(confidence_count);

    for (int c = 0; c < maskmaps_count; c++) {
        maskmaps[c] = (( float )maskmaps_u8[c] - ( float )maskmaps_zero_point) * maskmaps_scale;
    }

    for (int c = 0; c < location_count; c++) {
        location[c] = (( float )location_u8[c] - ( float )location_zero_point) * location_scale;
    }

    for (int c = 0; c < mask_count; c++) {
        mask[c] = (( float )mask_u8[c] - ( float )mask_zero_point) * mask_scale;
    }

    for (int c = 0; c < confidence_count; c++) {
        confidence[c] = exp((( float )confidence_u8[c] - ( float )confidence_zero_point) * confidence_scale);
    } 

    // if (VERBOSE) {
    //     std::cout << "Dequant and transform:" << get_current_time() - start << std::endl;   
    // }

    softmax(confidence);

    // if (VERBOSE) {
    //     std::cout << "Softmax:" << get_current_time() - start << std::endl;  
    // }

    /* postprocess */
    int num_class = 81;
    int num_priors = 18525;

    // if (VERBOSE) {
    //     std::cout << "Postprocessing:" << get_current_time() - start << std::endl;   
    // }   

    std::vector<Box2f> priorboxes = generate_priorbox(num_priors); 
    // const float confidence_thresh = 0.1f;
    const float nms_thresh = 0.3f;
    const int keep_top_k = 200;

    // if (VERBOSE) {
    //     std::cout << "Generate priorbox:" << get_current_time() - start << std::endl;
    // }

    std::vector<std::vector<Object>> class_candidates;
    class_candidates.resize(num_class);
    for (int i = 0; i < num_priors; i++) {
        const float* conf = confidence.data() + i * 81;
        const float* loc = location.data() + i * 4;
        const float* maskdata = mask.data() + i * 32;
        Box2f& priorbox = priorboxes[i];

        int label = 0;
        float score = 0.f;
        for (int j = 1; j < num_class; j++) {
            float class_score = conf[j];
            if (class_score > score) {
                label = j;
                score = class_score;
            }
        }

        if (label == 0 || score <= confidence_thresh)
            continue;

        float var[4] = {0.1f, 0.1f, 0.2f, 0.2f};

        float bbox_cx = var[0] * loc[0] * priorbox.w + priorbox.cx;
        float bbox_cy = var[1] * loc[1] * priorbox.h + priorbox.cy;
        float bbox_w = ( float )(exp(var[2] * loc[2]) * priorbox.w);
        float bbox_h = ( float )(exp(var[3] * loc[3]) * priorbox.h);

        float obj_x1 = bbox_cx - bbox_w * 0.5f;
        float obj_y1 = bbox_cy - bbox_h * 0.5f;
        float obj_x2 = bbox_cx + bbox_w * 0.5f;
        float obj_y2 = bbox_cy + bbox_h * 0.5f;

        obj_x1 = std::max(std::min(obj_x1 * bgr.cols, ( float )(bgr.cols - 1)), 0.f);
        obj_y1 = std::max(std::min(obj_y1 * bgr.rows, ( float )(bgr.rows - 1)), 0.f);
        obj_x2 = std::max(std::min(obj_x2 * bgr.cols, ( float )(bgr.cols - 1)), 0.f);
        obj_y2 = std::max(std::min(obj_y2 * bgr.rows, ( float )(bgr.rows - 1)), 0.f);

        Object obj;
        obj.rect = cv::Rect_<float>(obj_x1, obj_y1, obj_x2 - obj_x1 + 1, obj_y2 - obj_y1 + 1);
        obj.label = label;
        obj.prob = score;

        obj.maskdata = std::vector<float>(maskdata, maskdata + 32);

        class_candidates[label].push_back(obj);
    }

    // if (VERBOSE) {
    //     std::cout << "Find candidates:" << get_current_time() - start << std::endl;
    // }

    objects.clear();
    fast_nms(class_candidates, objects, nms_thresh, 0, keep_top_k);

    // if (VERBOSE) {
    //     std::cout << "NMS:" << get_current_time() - start << std::endl;
    // }

    for (std::size_t i = 0; i < objects.size(); i++) {
        Object& obj = objects[i];

        cv::Mat mask1(136, 136, CV_32FC1);
        mask1 = cv::Scalar(0.f);

        for (int p = 0; p < 32; p++) {
            const float* maskmap = maskmaps.data() + p;
            float coeff = obj.maskdata[p];
            float* mp = ( float* )mask1.data;

            // mask += m * coeff
            for (int j = 0; j < 136 * 136; j++) {
                mp[j] += maskmap[j * 32] * coeff;
            }
        }


        cv::Mat mask2;
        cv::resize(mask1, mask2, cv::Size(img_w, img_h));

        // crop obj box and binarize
        obj.mask = cv::Mat(img_h, img_w, CV_8UC1);      
        obj.mask = cv::Scalar(0);

        for (int y = 0; y < img_h; y++) {
            if (y < obj.rect.y - OFFSET || y > obj.rect.y - OFFSET + obj.rect.height)
                continue;

            const float* mp2 = mask2.ptr<const float>(y);
            uchar* bmp = obj.mask.ptr<uchar>(y);

            for (int x = 0; x < img_w; x++) {
                if (x < obj.rect.x - OFFSET || x > obj.rect.x - OFFSET + obj.rect.width)
                    continue;

                bmp[x] = mp2[x] > 0.5f ? 255 : 0;
            }
        }
        
    }
    
    // if (VERBOSE) {
    //     std::cout << "Mask operations:" << get_current_time() - start << std::endl;
    // }

    return 0;
}

static void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects, float thresh) {   
    const char* class_names[] = {"background",
                            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                            "train", "truck", "boat", "traffic light", "fire hydrant",
                            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                            "skis", "snowboard", "sports ball", "kite", "baseball bat",
                            "baseball glove", "skateboard", "surfboard", "tennis racket",
                            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                            "hot dog", "pizza", "donut", "cake", "chair", "couch",
                            "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                            "toaster", "sink", "refrigerator", "book", "clock", "vase",
                            "scissors", "teddy bear", "hair drier", "toothbrush"};

    static const unsigned char colors[81][3] = {
        {56, 0, 255},
        {226, 255, 0},
        {0, 94, 255},
        {0, 37, 255},
        {0, 255, 94},
        {255, 226, 0},
        {0, 18, 255},
        {255, 151, 0},
        {170, 0, 255},
        {0, 255, 56},
        {255, 0, 75},
        {0, 75, 255},
        {0, 255, 169},
        {255, 0, 207},
        {75, 255, 0},
        {207, 0, 255},
        {37, 0, 255},
        {0, 207, 255},
        {94, 0, 255},
        {0, 255, 113},
        {255, 18, 0},
        {255, 0, 56},
        {18, 0, 255},
        {0, 255, 226},
        {170, 255, 0},
        {255, 0, 245},
        {151, 255, 0},
        {132, 255, 0},
        {75, 0, 255},
        {151, 0, 255},
        {0, 151, 255},
        {132, 0, 255},
        {0, 255, 245},
        {255, 132, 0},
        {226, 0, 255},
        {255, 37, 0},
        {207, 255, 0},
        {0, 255, 207},
        {94, 255, 0},
        {0, 226, 255},
        {56, 255, 0},
        {255, 94, 0},
        {255, 113, 0},
        {0, 132, 255},
        {255, 0, 132},
        {255, 170, 0},
        {255, 0, 188},
        {113, 255, 0},
        {245, 0, 255},
        {113, 0, 255},
        {255, 188, 0},
        {0, 113, 255},
        {255, 0, 0},
        {0, 56, 255},
        {255, 0, 113},
        {0, 255, 188},
        {255, 0, 94},
        {255, 0, 18},
        {18, 255, 0},
        {0, 255, 132},
        {0, 188, 255},
        {0, 245, 255},
        {0, 169, 255},
        {37, 255, 0},
        {255, 0, 151},
        {188, 0, 255},
        {0, 255, 37},
        {0, 255, 0},
        {255, 0, 170},
        {255, 0, 37},
        {255, 75, 0},
        {0, 0, 255},
        {255, 207, 0},
        {255, 0, 226},
        {255, 245, 0},
        {188, 255, 0},
        {0, 255, 18},
        {0, 255, 75},
        {0, 255, 151},
        {255, 56, 0},
        {245, 255, 0}
    };

    cv::Mat &image = bgr;

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        if (obj.prob < thresh)
            continue;

        if (VERBOSE) {
            fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob, obj.rect.x, obj.rect.y,
                obj.rect.width, obj.rect.height);
        }

        const unsigned char* color = colors[color_index % 81];
        color_index++;

        cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));

        // draw mask
        for (int y = 0; y < image.rows; y++) {
            const uchar* mp = obj.mask.ptr(y);
            if (CORRECT_MASK) mp -= OFFSET + image.cols*OFFSET; // move pointer to correct masks
            uchar* p = image.ptr(y);
            for (int x = 0; x < image.cols; x++) {
                if (mp[x] == 255) {
                    p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                }
                p += 3;
            }
        }
    }
}

Model::Model(const char* model_file,  Frames* frames_) {
    init_tengine();
    frames = frames_;
    context_t context = create_context("timvx", 1);
    if (set_context_device(context, "TIMVX", NULL, 0))
        std::cout << "Set context failure!";
    graph = create_graph(context, "tengine", model_file);
    if (set_graph(544, 544, graph))
        std::cout << "Set graph failure!";
    input_tensor = get_graph_input_tensor(graph, 0, 0);
    std::cout << "Model created" << std::endl;
}

int Model::inference() {
    std::vector<Object> objects;
    // double start = get_current_time();   
    detect_yolact(frames->img, objects, graph, input_tensor, thresh);
    // if (VERBOSE) {
    //     std::cout << "Detect for: " << get_current_time() - start << std::endl;
    // }
    // start = get_current_time(); 
    draw_objects(frames->img, objects, thresh);
    // if (VERBOSE) {
    //     std::cout << "Draw objects for: " << get_current_time() - start << std::endl;
    // }
    frames->status = inferenced;

    return 0;
}

static unsigned long long min(Frames* frames) {
    if (frames[0].frame_num < frames[1].frame_num)
        return frames[0].frame_num;
    else 
        return frames[1].frame_num; 
}

int yolact() {
    unsigned long long counter = 0;
    unsigned long long showed = 0;
    Frames frames[2];
    cv::VideoCapture cap(0);

    if(!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat img;
    cap.read(img);
    Model yolact1("yolact_50_nosoft_KL_uint8.tmfile", &frames[0]);
    std::thread t1(yolact1);
    t1.detach();

    Model yolact2("yolact_50_nosoft_KL_uint8.tmfile", &frames[1]);
    std::thread t2(yolact2);
    t2.detach();

    // double start = get_current_time(); 
    while (true) {     
        cap.read(img);
        if (img.empty()) {
            std::cerr << "\nERROR! blank frame grabbed\n";
            release_tengine();
            return 0;
        }

        for (auto& frame: frames) {
            if (frame.status == inferenced && frame.frame_num == min(frames)) {
                cv::imshow("frame", frame.img);
                cv::waitKey(1);
                showed++;
                frame.frame_num = UINT64_MAX;               
                frame.status = waiting; 
                // if (showed%15==0) {
                //     std::cout << "\r" << "Fps: " << (15 / (get_current_time() - start)) * 1000 << std::flush;
                //     start = get_current_time();
                // }
            }
        }

        for (auto& frame: frames) {
            if (frame.status == waiting) {
                frame.img = img.clone();
                frame.frame_num = counter;
                frame.status = inferencing;
                break;
            }
        }
        counter++;
    }
}