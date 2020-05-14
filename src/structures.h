
#ifndef STRUCTURES_H
#define STRUCTURES_H

#define GOOGLE_CUDA 1

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <QColor>
#include <QDir>
#include <QString>
#include <QVector>

#include "../../ConfigReader/src/configreader.h"
#include "../../IncludeSpdlog/spdlog.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
//#include "rect.h"

constexpr auto NAME{ "Name" };

constexpr auto PRE_PROCESSING{ "PreProcessing" };
constexpr auto POST_PROCESSING{ "PostProcessing" };
constexpr auto BACKGROUND{ "BackgroundSubtractor" };

constexpr auto GENERAL{ "General" };
constexpr auto LOG_LEVEL{ "LogLevel" };

constexpr auto DATASET{ "Dataset" };

constexpr auto TENSORFLOW{ "TensorFlow" };
constexpr auto GRAPH{ "Graph" };
constexpr auto LABELS{ "Labels" };

constexpr auto INPUT_WIDTH{ "inputWidth" };
constexpr auto IN_HEIGHT{ "InputHeight" };
constexpr auto IN_MEAN{ "InputMean" };
constexpr auto IN_STD{ "InputStd" };
constexpr auto IN_LAYER{ "InputLayer" };
constexpr auto OUT_LAYER{ "OutputLayer" };
constexpr auto SELF_TEST{ "SelfTest" };
constexpr auto ROOT_DIR{ "RootDir" };

struct topScore
{
  tensorflow::string label;
  int indice;
  float score;
};

struct allScore
{
  std::vector<tensorflow::string> labels;
  tensorflow::Tensor indices;
  tensorflow::Tensor scores;
};

struct allScoreData
{
  std::vector<tensorflow::string> labels;
  std::vector<int> indices;
  std::vector<float> scores;
};

#endif // STRUCTURES_H
