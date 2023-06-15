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

    init_pos.x = box.x + (box.width - 1) / 2.0;
    init_pos.y = box.y + (box.height - 1) / 2.0;

    target_sz_w_ = static_cast<float>(w);
    target_sz_h_ = static_cast<float>(h);

    base_target_sz_w_ = target_sz_w_;
    base_target_sz_h_ = target_sz_h_;
    pos_ = init_pos;

//    // create hanning window
//    calculateHann(cv::Size(response_sz, response_sz), hann_window_);

    // exemplar and search sizes
    float context = (target_sz_w_ + target_sz_h_) * cfg_.context_amout;
    z_sz_ = std::sqrt((target_sz_w_ + context) * (target_sz_h_ + context));
    x_sz_ = z_sz_ * cfg_.instance_sz / cfg_.exemplar_sz;

    // Calculate channel average
    channel_average = cv::mean(img);
    for (int i = 0 ; i < 3 ; ++i){
        channel_average[i] = std::floor(channel_average[i]);
    }

    cv::Mat exemplar_image_patch;
    exemplar_image_patch = get_subwindow(img, init_pos, cfg_.exemplar_sz, std::round(z_sz_), channel_average);

    temp_->infer(exemplar_image_patch);

    hostDataBuffer1 = temp_->output1.ptr<float>(0);
    hostDataBuffer2 = temp_->output2.ptr<float>(0);
    hostDataBuffer3 = temp_->output3.ptr<float>(0);
}


void SiamRPNTrackerTRT::update(const cv::Mat& img) {
    cv::Mat instance_patch;
    instance_patch = get_subwindow(img, pos_ , cfg_.instance_sz, std::round(x_sz_), channel_average);
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
            }
        }
    }

    auto max_itr = std::max_element(response_vec.begin(), response_vec.end());
    auto id = std::distance(response_vec.begin(), max_itr);

    float offset_x = reg_vec.at(id).x * z_sz_ / cfg_.exemplar_sz;
    float offset_y = reg_vec.at(id).y * z_sz_ / cfg_.exemplar_sz;
    float offset_w = reg_vec.at(id).w * z_sz_ / cfg_.exemplar_sz;
    float offset_h = reg_vec.at(id).h * z_sz_ / cfg_.exemplar_sz;

    float lr = response_vec.at(id) * cfg_.lr;

    best_score = response[id];

    if (best_score > cfg_.confidence_score){
        pos_.x += offset_x;
        pos_.y += offset_y;

        target_sz_w_ = (1 - lr) * target_sz_w_ + lr * offset_w;
        target_sz_h_ = (1 - lr) * target_sz_h_ + lr * offset_h;

    }
    pos_.x = std::max(pos_.x, 0.f);
    pos_.y = std::max(pos_.y, 0.f);
    pos_.x = std::min(pos_.x, img.cols - 1.f);
    pos_.y = std::min(pos_.y, img.rows - 1.f);

    target_sz_w_ = std::max(target_sz_w_, 10.f);
    target_sz_h_ = std::max(target_sz_h_, 10.f);
    target_sz_w_ = std::min(target_sz_w_, img.cols - 1.f);
    target_sz_h_ = std::min(target_sz_h_, img.rows - 1.f);


    // # update exemplar and instance sizes
    float context = (target_sz_w_ + target_sz_h_) / 2.0;
    z_sz_ = std::sqrt((target_sz_w_ + context) * (target_sz_h_ + context));
    x_sz_ = z_sz_ * cfg_.instance_sz / cfg_.exemplar_sz;

    bbox.x = pos_.x - target_sz_w_ / 2;
    bbox.y =  pos_.y - target_sz_h_ / 2;
    bbox.width = target_sz_w_;
    bbox.height = target_sz_h_;

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

cv::Mat SiamRPNTrackerTRT::get_subwindow(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz, cv::Scalar avg_chans){
    int sz = original_sz;
    cv::Size im_sz = im.size();
    float center = (original_sz + 1) / 2.0;
    float context_xmin = std::floor(pos.x - center + 0.5);
    float context_xmax = context_xmin + sz - 1;
    float context_ymin = std::floor(pos.y - center + 0.5);
    float context_ymax = context_ymin + sz - 1;
    int left_pad = std::max(0, static_cast<int>(-context_xmin));
    int top_pad = std::max(0, static_cast<int>(-context_ymin));
    int right_pad = std::max(0, static_cast<int>(context_xmax - im_sz.width + 1));
    int bottom_pad = std::max(0, static_cast<int>(context_ymax - im_sz.height + 1));

    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;

    int r = im.rows;
    int c = im.cols;
    int k = im.channels();
    cv::Mat im_patch;

    if (top_pad || bottom_pad || left_pad || right_pad) {

        cv::Size size(c + left_pad + right_pad, r + top_pad + bottom_pad);
        cv::Mat te_im(size, CV_8UC(k), cv::Scalar(0, 0, 0));
        im.copyTo(te_im(cv::Rect(left_pad, top_pad, c, r)));
        if (top_pad) {
            te_im(cv::Rect(left_pad, 0, c, top_pad)) = avg_chans;
        }
        if (bottom_pad) {
            te_im(cv::Rect(left_pad, r + top_pad, c, te_im.rows - (r + top_pad))) = avg_chans;
        }
        if (left_pad) {
            te_im(cv::Rect(0, 0, left_pad, te_im.rows)) = avg_chans;
        }
        if (right_pad) {
            te_im(cv::Rect(c + left_pad, 0, te_im.cols - (c + left_pad), te_im.rows)) = avg_chans;
        }

        cv::Rect roi_rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1);
        im_patch = te_im(roi_rect);
    }
    else {
        im_patch = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }

    if (model_sz != original_sz) {
        cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz));
    }

    return im_patch;
}
