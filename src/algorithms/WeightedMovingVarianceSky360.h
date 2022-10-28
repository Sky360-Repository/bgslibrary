#pragma once

#include "IBGS.h"

namespace bgslibrary
{
    namespace algorithms
    {
        struct WeightedMovingVarianceParams
        {
            WeightedMovingVarianceParams(bool _enableWeight,
                                        bool _enableThreshold,
                                        int _threshold)
                : enableWeight(_enableWeight),
                enableThreshold(_enableThreshold),
                threshold(_threshold)
            {}

            bool enableWeight;
            bool enableThreshold;
            int threshold;
        };

        class WeightedMovingVarianceSky360 
            : public IBGS {
        public:
            static const bool DEFAULT_ENABLE_WEIGHT{true};
            static const bool DEFAULT_ENABLE_THRESHOLD{true};
            static const int DEFAULT_THRESHOLD_VALUE{15};

            WeightedMovingVarianceSky360(bool _enableWeight = DEFAULT_ENABLE_WEIGHT,
                                    bool _enableThreshold = DEFAULT_ENABLE_THRESHOLD,
                                    int _threshold = DEFAULT_THRESHOLD_VALUE);
            ~WeightedMovingVarianceSky360();

            cv::Mat getBackgroundModel();

            void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

        private:
            static inline const float ONE_THIRD{1.0f / 3.0f};

            const int m_numProcessesParallel;
            std::vector<int> m_processSeq;
            std::vector<std::array<std::unique_ptr<cv::Mat>, 2>> imgInputPrevParallel;

            WeightedMovingVarianceParams m_params;

            void processParallel(const cv::Mat &_imgInput, cv::Mat &_imgOutput);
            static void process(const cv::Mat &img_input, 
                                cv::Mat &img_output, 
                                std::array<std::unique_ptr<cv::Mat>, 2>& img_input_prev, 
                                const WeightedMovingVarianceParams& _params);
            static void weightedVarianceMono(
                    const cv::Mat &_img1, 
                    const cv::Mat &_img2, 
                    const cv::Mat &_img3, 
                    cv::Mat& _outImg,
                    const WeightedMovingVarianceParams& _params);
            static void weightedVarianceColor(
                    const cv::Mat &_img1, 
                    const cv::Mat &_img2, 
                    const cv::Mat &_img3, 
                    cv::Mat& _outImg,
                    const WeightedMovingVarianceParams& _params);

            void save_config(cv::FileStorage &fs);
            void load_config(cv::FileStorage &fs);                    
        };

        bgs_register(WeightedMovingVarianceSky360);
    }
}