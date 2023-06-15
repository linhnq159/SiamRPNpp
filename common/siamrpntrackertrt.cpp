#include "siamrpntrackertrt.h"

SiamRPNTrackerTRT::SiamRPNTrackerTRT(TRTTemplate* temp, TRTTrack* track)
{
    temp_ = temp;
    track_ = track;
    // Creat Anchors

    int response_sz = (cfg_.instance_sz - cfg_.exemplar_sz) / cfg_.total_stride + 1 + cfg_.base_size;
    createAnchors(response_sz);

    // create hanning window
    calculateHann(cv::Size(response_sz, response_sz), hann_window_);
}

void SiamRPNTrackerTRT::createAnchors(const int& response_sz) {
    float ratios[5] = {0.33, 0.5, 1, 2, 3};
    float scales = 8;
    int total_stride = 8;
    int anchor_num = 5;
    int size = total_stride * total_stride;
    int total_anchor_num = response_sz * response_sz * anchor_num;
    std::vector<Anchor> arr_anchor;

    for (int i = 0; i < 5; ++i) {
        int w = static_cast<int>(std::sqrt(size / ratios[i]));
        int h = static_cast<int>(w * ratios[i]);
        Anchor anchor_temp;
        anchor_temp.x = 0;
        anchor_temp.y = 0;
        anchor_temp.w = w * scales;
        anchor_temp.h = h * scales;
        arr_anchor.push_back(anchor_temp);
    }

    float beg_offset = -(response_sz / 2) * total_stride;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < response_sz; ++j) {
            for (int k = 0; k < response_sz; ++k) {
                float xs = beg_offset + total_stride * k;
                float ys = beg_offset + total_stride * j;
                Anchor anchor_temp;
                anchor_temp.x = xs;
                anchor_temp.y = ys;
                anchor_temp.w = arr_anchor[i].w;
                anchor_temp.h = arr_anchor[i].h;
                anchors_.push_back(anchor_temp);
            }
        }
    }
}

void SiamRPNTrackerTRT::calculateHann(const cv::Size& sz, cv::Mat& output) {
    cv::Mat temp1(cv::Size(sz.width, 1), CV_32FC1);
    cv::Mat temp2(cv::Size(sz.height, 1), CV_32FC1);

    float* p1 = temp1.ptr<float>(0);
    float* p2 = temp2.ptr<float>(0);

    for (int i = 0; i < sz.width; ++i) p1[i] = 0.5 * (1 - cos(CV_2PI * i / (sz.width - 1)));

    for (int i = 0; i < sz.height; ++i) p2[i] = 0.5 * (1 - cos(CV_2PI * i / (sz.height - 1)));

    output = temp2.t() * temp1;
}

void SiamRPNTrackerTRT::init(const cv::Mat& img, cv::Rect2d& box) {
    bbox = box ;

    double w = box.width;
    double h = box.height;

    cv::Point2f init_pos;

    init_pos.x = box.x + box.width / 2.0;
    init_pos.y = box.y + box.height / 2.0;

    target_sz_w_ = static_cast<float>(w);
    target_sz_h_ = static_cast<float>(h);

    base_target_sz_w_ = target_sz_w_;
    base_target_sz_h_ = target_sz_h_;
    pos_ = init_pos;

//    int response_sz = (cfg_.instance_sz - cfg_.exemplar_sz) / cfg_.total_stride + 1 + cfg_.base_size;
//    createAnchors(response_sz);

//    // create hanning window
//    calculateHann(cv::Size(response_sz, response_sz), hann_window_);

    // exemplar and search sizes
    float context = (target_sz_w_ + target_sz_h_) * cfg_.context_amout;
    z_sz_ = std::sqrt((target_sz_w_ + context) * (target_sz_h_ + context));
    x_sz_ = z_sz_ * cfg_.instance_sz / cfg_.exemplar_sz;

    cv::Mat exemplar_image_patch;
    exemplar_image_patch = getSamplePatch(img, init_pos, z_sz_, cfg_.exemplar_sz);

//    std::cout << "exemplar_image_patch_float" << exemplar_image_patch << std::endl;
//    cv::Size size = exemplar_image_patch.size();
//    int rows = size.height;
//    int cols = size.width;
//    std::cout << "Rows: " << rows << ", Cols: " << cols << std::endl;

    temp_->infer(exemplar_image_patch);

    hostDataBuffer1 = temp_->output1.ptr<float>(0);
    hostDataBuffer2 = temp_->output2.ptr<float>(0);
    hostDataBuffer3 = temp_->output3.ptr<float>(0);

}


void SiamRPNTrackerTRT::update(const cv::Mat& img) {
    cv::Mat instance_patch;
    instance_patch = getSamplePatch(img, pos_, x_sz_, cfg_.instance_sz);
    track_->infer(instance_patch, hostDataBuffer1, hostDataBuffer2, hostDataBuffer3);

    float* output_cls_ = track_->output_cls.ptr<float>(0);
    float* output_reg_ = track_->output_reg.ptr<float>(0);

    std::vector<Anchor> reg_vec;
    reg_vec.resize(anchors_.size());

    for (size_t i = 0; i != anchors_.size(); ++i) {
        reg_vec.at(i).x = output_reg_[i] * anchors_.at(i).w + anchors_.at(i).x;
        reg_vec.at(i).y = output_reg_[1 * anchors_.size() + i] * anchors_.at(i).h + anchors_.at(i).y;
        reg_vec.at(i).w = std::exp(output_reg_[2*anchors_.size() + i]) * anchors_.at(i).w;
        reg_vec.at(i).h = std::exp(output_reg_[3*anchors_.size() + i]) * anchors_.at(i).h;
    }


    // set penalty
    std::vector<float> penalty = createPenalty(target_sz_w_, target_sz_h_, reg_vec);

    // get response score
    // Soft max output cls
    std::vector<float> response;
    for (size_t i = 0; i != anchors_.size() ; i++){
        response.push_back(std::exp(output_cls_[anchors_.size() + i]) / (std::exp(output_cls_[i]) + std::exp(output_cls_[anchors_.size() + i])));
    }

    std::vector<float> response_penalty;
    for (size_t i = 0; i < penalty.size(); ++i) {
        response_penalty.emplace_back(response[i] * penalty[i]);
        // cout << "response_penalty: " << response_penalty.device() << endl;
    }

    // TODO:: response transfer to vector
    std::vector<float> response_vec;
    for (int n = 0; n < ANCH_NUM; ++n) {
        for (int r = 0; r < hann_window_.rows; ++r) {
            float* phann = hann_window_.ptr<float>(r);
            for (int c = 0; c < hann_window_.cols; ++c) {
                float temp =
                    (1 - cfg_.win_influence) *
                        response_penalty[n * hann_window_.cols * hann_window_.rows + r * hann_window_.cols + c] +
                    cfg_.win_influence * phann[c];
                response_vec.push_back(temp);
                // cout << "response_vec: " << response_vec.device() << endl;
            }
        }
    }

    auto max_itr = std::max_element(response_vec.begin(), response_vec.end());
    auto id = std::distance(response_vec.begin(), max_itr);

    float offset_x = reg_vec.at(id).x * z_sz_ / cfg_.exemplar_sz;
    float offset_y = reg_vec.at(id).y * z_sz_ / cfg_.exemplar_sz;
    float offset_w = reg_vec.at(id).w * z_sz_ / cfg_.exemplar_sz;
    float offset_h = reg_vec.at(id).h * z_sz_ / cfg_.exemplar_sz;

    pos_.x += offset_x;
    pos_.y += offset_y;
    pos_.x = std::max(pos_.x, 0.f);
    pos_.y = std::max(pos_.y, 0.f);
    pos_.x = std::min(pos_.x, img.cols - 1.f);
    pos_.y = std::min(pos_.y, img.rows - 1.f);

    float lr = response_vec.at(id) * cfg_.lr;
    target_sz_w_ = (1 - lr) * target_sz_w_ + lr * offset_w;
    target_sz_h_ = (1 - lr) * target_sz_h_ + lr * offset_h;

    target_sz_w_ = std::max(target_sz_w_, 10.f);
    target_sz_h_ = std::max(target_sz_h_, 10.f);
    target_sz_w_ = std::min(target_sz_w_, img.cols - 1.f);
    target_sz_h_ = std::min(target_sz_h_, img.rows - 1.f);

    // # update exemplar and instance sizes
    float context = (target_sz_w_ + target_sz_h_) / 2.0;
    z_sz_ = std::sqrt((target_sz_w_ + context) * (target_sz_h_ + context));
    x_sz_ = z_sz_ * cfg_.instance_sz / cfg_.exemplar_sz;

    bbox.x = pos_.x + 1 - (target_sz_w_ - 1) / 2;
    bbox.y =  pos_.y + 1 - (target_sz_h_ - 1) / 2;
    bbox.width = target_sz_w_;
    bbox.height = target_sz_h_;

    // box.x = 20 ;  //pos_.x + 1 - (target_sz_w_ - 1) / 2;
    // box.y = 20 ;// pos_.y + 1 - (target_sz_h_ - 1) / 2;
    // box.width = 10;// target_sz_w_;
    // box.height = 10; //target_sz_h_;

}

std::vector<float> SiamRPNTrackerTRT::createPenalty(const float& target_w, const float& target_h,
                                                 const std::vector<Anchor>& offsets) {
    std::vector<float> result;

    auto padded_sz = [](const float& w, const float& h) {
        float context_tmp = 0.5 * (w + h);
        return std::sqrt((w + context_tmp) * (h + context_tmp));
    };
    auto larger_ratio = [](const float& r) { return std::max(r, 1 / r); };
    for (size_t i = 0; i < offsets.size(); ++i) {
        auto src_sz = padded_sz(target_w * cfg_.exemplar_sz / z_sz_, target_h * cfg_.exemplar_sz / z_sz_);
        auto dst_sz = padded_sz(offsets[i].w, offsets[i].h);
        auto change_sz = larger_ratio(dst_sz / src_sz);

        float src_ratio = target_w / target_h;
        float dst_ratio = offsets[i].w / offsets[i].h;
        float change_ratio = larger_ratio(dst_ratio / src_ratio);
        result.emplace_back(std::exp(-(change_ratio * change_sz - 1) * cfg_.penalty_k));
    }
    return result;
}

cv::Mat subwindowtrt(const cv::Mat& in, const cv::Rect& window, int borderType) {
    cv::Rect cutWindow = window;
    limit(cutWindow, in.cols, in.rows);

    if (cutWindow.height <= 0 || cutWindow.width <= 0) assert(0);

    cv::Rect border = getBorder(window, cutWindow);
    cv::Mat res = in(cutWindow);

    if (border != cv::Rect(0, 0, 0, 0)) {
        cv::copyMakeBorder(res, res, border.y, border.height, border.x, border.width, borderType);
    }
    return res;
}

cv::Mat SiamRPNTrackerTRT::getSamplePatch(const cv::Mat im, const cv::Point2f posf, const int& in_sz, const int& out_sz) {
    // Pos should be integer when input, but floor in just in case.
    cv::Point2i pos(posf.x, posf.y);
    cv::Size sample_sz = {in_sz, in_sz};  // scale adaptation
    cv::Size model_sz = {out_sz, out_sz};

    // Downsample factor
    float resize_factor = std::min(sample_sz.width / out_sz, sample_sz.height / out_sz);
    int df = std::max((float)std::floor(resize_factor - 0.1), float(1));

    cv::Mat new_im;
    im.copyTo(new_im);
    if (df > 1) {
        // compute offset and new center position
        cv::Point os((pos.x - 1) % df, ((pos.y - 1) % df));
        pos.x = (pos.x - os.x - 1) / df + 1;
        pos.y = (pos.y - os.y - 1) / df + 1;
        // new sample size
        sample_sz.width = sample_sz.width / df;
        sample_sz.height = sample_sz.height / df;
        // down sample image
        int r = (im.rows - os.y) / df + 1;
        int c = (im.cols - os.x) / df;
        cv::Mat new_im2(r, c, im.type());
        new_im = new_im2;
        for (size_t i = 0 + os.y, m = 0; i < (size_t)im.rows && m < (size_t)new_im.rows; i += df, ++m) {
            for (size_t j = 0 + os.x, n = 0; j < (size_t)im.cols && n < (size_t)new_im.cols; j += df, ++n) {
                if (im.channels() == 1) {
                    new_im.at<uchar>(m, n) = im.at<uchar>(i, j);
                } else {
                    new_im.at<cv::Vec3b>(m, n) = im.at<cv::Vec3b>(i, j);
                }
            }
        }
    }

    // make sure the size is not too small and round it
    sample_sz.width = std::max(std::round(sample_sz.width), 2.0);
    sample_sz.height = std::max(std::round(sample_sz.height), 2.0);

    cv::Point pos2(pos.x - std::floor((sample_sz.width + 1) / 2), pos.y - std::floor((sample_sz.height + 1) / 2));
    cv::Mat im_patch = subwindowtrt(new_im, cv::Rect(pos2, sample_sz), cv::BORDER_REPLICATE);

    cv::Mat resized_patch;
    if (im_patch.cols == 0 || im_patch.rows == 0) {
        return resized_patch;
    }
    cv::resize(im_patch, resized_patch, model_sz);

    return resized_patch;
}
