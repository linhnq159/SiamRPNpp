#include "trttrack.h"


bool TRTTrack::CLoadEngineTrack(char *TrtPath)
{
    std::ifstream ifile(TrtPath, std::ios::in | std::ios::binary);
    if (!ifile)
    {
        std::cout << "model file: " << TrtPath << " not found!" << std::endl;
        return false;
    }
    ifile.seekg(0, std::ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    std::vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine((void *)&buf[0], mdsize, nullptr));

    // Context
    context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Get size Input Output
    mInputDimsTrack = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.inputTensorNamesTrack[0].c_str()));
    mInputDimsTrack1 = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.inputTensorNamesTrack[1].c_str()));

    mOutputDimsTrack1 = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.outputTensorNamesTrack[0].c_str()));
    mOutputDimsTrack2 = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.outputTensorNamesTrack[1].c_str()));

    return true;
}

bool TRTTrack::infer(cv::Mat& frame, float* hostDataBuffer1, float* hostDataBuffer2, float* hostDataBuffer3)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

//    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
//    if (!context)
//    {
//        return false;
//    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNamesTrack.size() == 4);

    if (!processInput(buffers, frame, hostDataBuffer1, hostDataBuffer2, hostDataBuffer3))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!

bool TRTTrack::processInput(const samplesCommon::BufferManager& buffers, cv::Mat& frame, float* input1, float* input2, float* input3)
{
    // size input
    const int inputC = mInputDimsTrack.d[1];
    const int inputH = mInputDimsTrack.d[2];
    const int inputW = mInputDimsTrack.d[3];

    // size input1
    const int inputC1 = mInputDimsTrack1.d[1];
    const int inputH1 = mInputDimsTrack1.d[2];
    const int inputW1 = mInputDimsTrack1.d[3];

    // Convert cv::Mat to tensor float 1x3x127x127
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNamesTrack[0]));
    float* hostDataBuffer1 = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNamesTrack[1]));
    float* hostDataBuffer2 = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNamesTrack[2]));
    float* hostDataBuffer3 = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNamesTrack[3]));

    // Add input to buffers
    for (int c = 0; c < inputC; ++c){
        for (int h = 0; h < inputH; ++h){
            for (int w = 0; w < inputW; ++w){
                int dstIdx = c * inputH * inputW + h * inputW + w;
                hostDataBuffer[dstIdx] = frame.at<cv::Vec3b>(h, w)[c];
            }
        }
    }

    for (int c = 0; c < inputC1; ++c) {
        for (int h = 0; h < inputH1; ++h) {
            for (int w = 0; w < inputW1; ++w) {
                int index = c * inputH1 * inputW1 + h * inputW1 + w;
                hostDataBuffer1[index] = input1[index];
                hostDataBuffer2[index] = input2[index];
                hostDataBuffer3[index] = input3[index];
            }
        }
    }
    return true;
}

bool TRTTrack::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    float *output_cls = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNamesTrack[0]));
    float *output_reg = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNamesTrack[1]));

//    for (int i = 0 ; i < 20 ; ++i)
//        std::cout << "output_cls[i] :" << output_cls[i] << std::endl;


    // Size *output
    int channels_cls = mOutputDimsTrack1.d[1];
    int height_cls = mOutputDimsTrack1.d[2];
    int width_cls = mOutputDimsTrack1.d[3];

    int channels_reg = mOutputDimsTrack2.d[1];
    int height_reg = mOutputDimsTrack2.d[2];
    int width_reg = mOutputDimsTrack2.d[3];

    int outputSizeCls = channels_cls * height_cls * width_cls;
    int outputSizeReg = channels_reg * height_reg * width_reg;

    this->output_cls = cv::Mat(1,outputSizeCls,CV_32F);
    this->output_reg = cv::Mat(1,outputSizeReg,CV_32F);
    //Cach lay con tron output float*
//    this->output1.ptr<float>(0);
    cv::Mat(1,outputSizeCls,CV_32F,output_cls).copyTo(this->output_cls);
    cv::Mat(1,outputSizeReg,CV_32F,output_reg).copyTo(this->output_reg);


    return true;
}
