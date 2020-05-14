#ifndef TENSORGRAPH_H
#define TENSORGRAPH_H

#include <QJsonObject>
#include <fstream>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "structures.h"

class TensorGraph : public QObject
{
  Q_OBJECT
 public:
  TensorGraph(QJsonObject const &a_config);
  // virtual ~TensorGraph();
  int configRead();
  void initPreprocess(std::string name);

  void initInception();
  void initMobilnetROI();
  void initMobilnet();
  int get_label_count() { return m_how_many_labels; }
  bool get_graph_exist() { return m_graph_exist; }

  tensorflow::Status LoadGraph(const tensorflow::string &graph_file_name,
                               std::unique_ptr<tensorflow::Session> *session);
  void initModel();
  void selectImage(cv::Mat &imageMat);

  tensorflow::Status GetTopLabels(const std::vector<tensorflow::Tensor> &outputs, tensorflow::Tensor *indices,
                                  tensorflow::Tensor *scores);

  tensorflow::Status ReadLabelsFile(const tensorflow::string &file_name, std::vector<tensorflow::string> *result,
                                    size_t *found_label_count);

  struct topScore returnTopLabel();
  struct allScoreData returnAllLabel();

 private:
  long sysID;
  std::shared_ptr<spdlog::logger> console;
  tensorflow::string image;
  tensorflow::string graph;
  tensorflow::string labels;

 private:
  tensorflow::string m_graph;
  tensorflow::string m_labels;
  qint32 m_inputWidth;
  qint32 m_inputHeight;
  qint32 m_inputMean;
  qint32 m_inputStd;
  tensorflow::string m_inputLayer;
  tensorflow::string m_outputLayer;
  bool m_selfTest;
  tensorflow::string m_rootDir;

  tensorflow::string m_graph_path;
  tensorflow::string m_labels_path;

  std::vector<tensorflow::string> m_labelsOutput;
  std::vector<tensorflow::Tensor> m_outputs;
  std::unique_ptr<tensorflow::Session> m_session;

  size_t m_label_count;
  int m_how_many_labels;

 private:
  tensorflow::string graph_path;
  tensorflow::string labels_path;

  tensorflow::int32 input_width;
  tensorflow::int32 input_height;
  float input_mean;
  float input_std;
  tensorflow::string input_layer;
  tensorflow::string output_layer;
  bool self_test;
  tensorflow::string root_dir;

  // struct s_config config;

 private:
  bool m_graph_exist{};
  bool m_text_exist{};

 signals:

 public slots:
  void onUpdate();
};

#endif /* TENSORGRAPH_H */
