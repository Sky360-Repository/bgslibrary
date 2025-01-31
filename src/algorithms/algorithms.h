#pragma once

#include "FrameDifference.h"
#include "StaticFrameDifference.h"
#include "WeightedMovingMean.h"
#include "WeightedMovingVariance.h"
#include "MixtureOfGaussianV1.h" // Only for OpenCV == 2
#include "MixtureOfGaussianV2.h"
#include "AdaptiveBackgroundLearning.h"
#include "AdaptiveSelectiveBackgroundLearning.h"
#include "KNN.h" // Only for OpenCV >= 3
#include "GMG.h" // Only for OpenCV >= 2.4.3 and OpenCV < 3
#include "DPAdaptiveMedian.h" // Only for OpenCV 2 or 3
#include "DPGrimsonGMM.h" // Only for OpenCV 2 or 3
#include "DPZivkovicAGMM.h" // Only for OpenCV 2 or 3
#include "DPMean.h" // Only for OpenCV 2 or 3
#include "DPWrenGA.h" // Only for OpenCV 2 or 3
#include "DPPratiMediod.h" // Only for OpenCV 2 or 3
#include "DPEigenbackground.h" // Only for OpenCV 2 or 3
#include "DPTexture.h" // Only for OpenCV 2 or 3
#include "T2FGMM_UM.h" // Only for OpenCV 2 or 3
#include "T2FGMM_UV.h" // Only for OpenCV 2 or 3
#include "T2FMRF_UM.h" // Only for OpenCV 2 or 3
#include "T2FMRF_UV.h" // Only for OpenCV 2 or 3
#include "FuzzySugenoIntegral.h" // Only for OpenCV 2 or 3
#include "FuzzyChoquetIntegral.h" // Only for OpenCV 2 or 3
#include "LBSimpleGaussian.h" // Only for OpenCV 2 or 3
#include "LBFuzzyGaussian.h" // Only for OpenCV 2 or 3
#include "LBMixtureOfGaussians.h" // Only for OpenCV 2 or 3
#include "LBAdaptiveSOM.h" // Only for OpenCV 2 or 3
#include "LBFuzzyAdaptiveSOM.h" // Only for OpenCV 2 or 3
#include "LBP_MRF.h" // Only for OpenCV 2 or OpenCV <= 3.4.7
#include "MultiLayer.h" // Only for OpenCV 2 or OpenCV <= 3.4.7
#include "PixelBasedAdaptiveSegmenter.h"
#include "VuMeter.h" // Only for OpenCV 2 or 3
#include "KDE.h" // Only for OpenCV 2 or 3
#include "IndependentMultimodal.h" // Only for OpenCV 2 or 3
#include "MultiCue.h" // Only for OpenCV 2 or 3
#include "SigmaDelta.h"
#include "SuBSENSE.h"
#include "LOBSTER.h"
#include "PAWCS.h"
#include "TwoPoints.h"
#include "ViBe.h"
#include "CodeBook.h"
#include "VibeBGS.h"
#include "WeightedMovingVarianceSky360.h"

//#include "_template_/MyBGS.h"

using namespace bgslibrary::algorithms;
