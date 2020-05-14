#include "tensorgraph.h"
#include <QFile>

TensorGraph::TensorGraph(QJsonObject const &a_config)
  : m_graph_exist(false)
  , m_text_exist(false)
{
  m_graph = a_config[GRAPH].toString().toStdString();
  m_labels = a_config[LABELS].toString().toStdString();

  QFile file(QString::fromStdString(m_graph));
  if (file.exists())
  {
    m_graph_exist = true;
  }
  QFile file2(QString::fromStdString(m_labels));
  if (file2.exists())
  {
    m_text_exist = true;
  }

  if (m_graph_exist && m_text_exist)
  {
    TensorGraph::initMobilnet();
    TensorGraph::initModel();
  }
}
void TensorGraph::onUpdate() {}

void TensorGraph::initPreprocess(std::string name) {}

int TensorGraph::configRead() {}

void TensorGraph::TensorGraph::initInception()
{
  m_inputWidth = 299;
  m_inputHeight = 299;
  m_inputMean = 0;
  m_inputStd = 255;
  m_inputLayer = "input";
  m_outputLayer = "InceptionV3/Predictions/Reshape_1";
  m_selfTest = false;
  m_rootDir = "";
}

void TensorGraph::initMobilnetROI()
{
  input_width = 224;
  input_height = 224;
  input_mean = 0;
  input_std = 224;
  input_layer = "input";
  output_layer = "final_result";
  self_test = false;
  root_dir = "";
}
void TensorGraph::initMobilnet()
{
  m_inputWidth = 224;
  m_inputHeight = 224;
  m_inputMean = 0;
  m_inputStd = 224;
  m_inputLayer = "input";
  m_outputLayer = "final_result";
  m_selfTest = false;
  m_rootDir = "";
}

tensorflow::Status TensorGraph::LoadGraph(const tensorflow::string &graph_file_name,
                                          std::unique_ptr<tensorflow::Session> *session)
{
  tensorflow::GraphDef graph_def;
  tensorflow::Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok())
  {
    H_Logger->error("Failed to load compute graph at {} ", graph_file_name);
    return tensorflow::Status::OK();
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  tensorflow::Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok())
  {
    return session_create_status;
  }
  return tensorflow::Status::OK();
}

static tensorflow::SessionOptions MakeSessionOptions(const tensorflow::string &visible_device_list = "",
                                                     double per_process_gpu_memory_fraction = 0.,
                                                     int gpu_device_count = 1,
                                                     const std::vector<std::vector<float>> &memory_limit_mb = {})
{
  tensorflow::SessionOptions options;
  tensorflow::ConfigProto *config = &options.config;
  (*config->mutable_device_count())["GPU"] = gpu_device_count;
  tensorflow::GPUOptions *gpu_options = config->mutable_gpu_options();
  gpu_options->set_visible_device_list(visible_device_list);
  gpu_options->set_per_process_gpu_memory_fraction(per_process_gpu_memory_fraction);
  for (const auto &v : memory_limit_mb)
  {
    auto virtual_devices = gpu_options->mutable_experimental()->add_virtual_devices();
    for (float mb : v)
    {
      virtual_devices->add_memory_limit_mb(mb);
    }
  }
  return options;
}

const char *kDeviceNamePrefix = "/job:localhost/replica:0/task:0";

void TensorGraph::initModel()
{
  setenv("CUDA_VISIBLE_DEVICES", "", 1);
  // tensorflow::SessionOptions opts =
  // MakeSessionOptions("CUDA_VISIBLE_DEVICES=0",0.1,1,{{500}});
  // std::vector<tensorflow::Device*> devices;
  // tensorflow::DeviceFactory::GetFactory("GPU")->CreateDevices(opts,
  // kDeviceNamePrefix, &devices); tensorflow::EXPECT_EQ(1, devices.size());

  // First we load and initialize the model.
  // std::unique_ptr<tensorflow::Session> session;
  // tensorflow::string _graph_path = m_graph.toStdString();
  // tensorflow::string _labels_path = m_labels.toStdString();
  H_Logger->trace("TensorGraph::initModel() m_graph:{}", m_graph);
  H_Logger->trace("TensorGraph::initModel() m_labels:{}", m_labels);

  tensorflow::Status load_graph_status = LoadGraph(m_graph, &m_session);
  if (!load_graph_status.ok())
  {
    LOG(ERROR) << load_graph_status;
    H_Logger->error("load_graph_status failed ");
  }

  tensorflow::Status read_labels_status = ReadLabelsFile(m_labels, &m_labelsOutput, &m_label_count);
  if (!read_labels_status.ok())
  {
    H_Logger->error("TensorGraph::initModel() read_labels_status:{}", read_labels_status.ok());
    LOG(ERROR) << read_labels_status;
    return;
  }
  H_Logger->debug("m_labelsOutput:");
  for (int i = 0; i < m_labelsOutput.size(); i++)
  {
    H_Logger->debug("[{}]:{}", i, m_labelsOutput[i]);
  }
  m_how_many_labels = (int)(m_label_count);
  H_Logger->debug("TensorGraph::initModel() m_how_many_labels:{}", m_how_many_labels);
}

void TensorGraph::selectImage(cv::Mat &imageMat)
{
  if (!imageMat.empty())
  {
    cv::Mat ImageProcess;
    cv::Size s(m_inputHeight, m_inputWidth);
    cv::resize(imageMat, ImageProcess, s, 0, 0, cv::INTER_CUBIC);
    H_Logger->trace("resize to({},{})", m_inputHeight, m_inputWidth);
    int depth = ImageProcess.channels();
    H_Logger->trace("TensorGraph::selectImage() depth:{}", depth);
    if (depth == 3 || depth == 4)
    {
      cv::cvtColor(ImageProcess, ImageProcess, 6);
      H_Logger->trace("TensorGraph::selectImage() cvtColor");
    }

    // creating a Tensor for storing the data
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({ 1, m_inputHeight, m_inputWidth, depth }));
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();

    cv::Mat ImageFloat;
    ImageProcess.convertTo(ImageFloat, CV_32FC1);
    ImageProcess = ImageFloat;
    ImageProcess = ImageProcess - m_inputMean;
    ImageProcess = ImageProcess / m_inputStd;
    const float *source_data = (float *)ImageProcess.data;
    // copying the data into the corresponding tensor
    for (int y = 0; y < m_inputHeight; ++y)
    {
      const float *source_row = source_data + (y * m_inputWidth * depth);
      for (int x = 0; x < m_inputWidth; ++x)
      {
        const float *source_pixel = source_row + (x * depth);
        for (int c = 0; c < depth; ++c)
        {
          const float *source_value = source_pixel + c;
          input_tensor_mapped(0, y, x, c) = *source_value;
        }
      }
    }
    H_Logger->trace("TensorGraph::selectImage() run_status");
    tensorflow::Status run_status =
        (m_session)->Run({ { m_inputLayer, input_tensor } }, { m_outputLayer }, {}, &m_outputs);
  }
  else
  {
    H_Logger->error("TensorGraph::selectImage() failed resize image");
  }
}

tensorflow::Status TensorGraph::GetTopLabels(const std::vector<tensorflow::Tensor> &outputs,
                                             tensorflow::Tensor *indices, tensorflow::Tensor *scores)
{
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops; // NOLINT(build/namespaces)

  tensorflow::string output_name = "top_k";
  TopK(root.WithOpName(output_name), m_outputs[0], m_how_many_labels);
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  std::vector<tensorflow::Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, { output_name + ":0", output_name + ":1" }, {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return tensorflow::Status::OK();
}

tensorflow::Status TensorGraph::ReadLabelsFile(const tensorflow::string &file_name,
                                               std::vector<tensorflow::string> *result, size_t *found_m_label_count)
{
  std::ifstream file(file_name);

  if (!file)
  {
    // return tensorflow::errors::NotFound("Labels file ", file_name," not
    // found.");
    H_Logger->error("TensorGraph:: error:Labels file {}", file_name);
    return tensorflow::Status::OK();
  }
  result->clear();
  tensorflow::string line;
  while (std::getline(file, line))
  {
    result->push_back(line);
  }
  *found_m_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding)
  {
    result->emplace_back();
  }
  return tensorflow::Status::OK();
}

struct topScore TensorGraph::returnTopLabel()
{
  struct topScore topScoreOutput;
  int _how_many_labels = std::min(3, (int)(m_label_count));
  tensorflow::Tensor indices;
  tensorflow::Tensor scores;
  tensorflow::Status getTopLabels_status = (GetTopLabels(m_outputs, &indices, &scores));
  if (!getTopLabels_status.ok())
  {
    H_Logger->error("getTopLabels_status failed!");
  }
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<tensorflow::int32>::Flat indices_flat = indices.flat<tensorflow::int32>();
  for (int pos = 0; pos < _how_many_labels; ++pos)
  {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    H_Logger->trace("TensorGraph:: {}({}):{}", m_labelsOutput[label_index], label_index, score);
  }

  topScoreOutput.indice = indices_flat(0);
  topScoreOutput.score = scores_flat(0);
  topScoreOutput.label = m_labelsOutput[topScoreOutput.indice];

  return topScoreOutput;
}

struct allScoreData TensorGraph::returnAllLabel()
{
  int _how_many_labels = (int)(m_label_count);
  tensorflow::Tensor indices;
  tensorflow::Tensor scores;

  tensorflow::Status getTopLabels_status = (GetTopLabels(m_outputs, &indices, &scores));
  if (!getTopLabels_status.ok())
  {
    H_Logger->trace("TensorGraph::returnAllLabel() failed");
  }
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<tensorflow::int32>::Flat indices_flat = indices.flat<tensorflow::int32>();

  struct allScoreData topScoreDataOutput;
  for (int tempI = 0; tempI < _how_many_labels; tempI++)
  {
    topScoreDataOutput.indices.push_back(0);
    topScoreDataOutput.scores.push_back(0.0);
    topScoreDataOutput.labels.push_back("0");
  }

  for (int pos = 0; pos < _how_many_labels; ++pos)
  {
    int label_index = indices_flat(pos); // indices_flat(pos);
    topScoreDataOutput.indices[label_index] = label_index;

    float score = scores_flat(pos); // scores_flat(pos);
    topScoreDataOutput.scores[label_index] = score;
    topScoreDataOutput.labels[label_index] = m_labelsOutput[label_index];
    H_Logger->trace("TensorGraph:: {}({}):{}", m_labelsOutput[label_index], label_index, score);
  }
  return topScoreDataOutput;
}
