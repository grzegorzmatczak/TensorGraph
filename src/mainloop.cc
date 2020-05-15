#include "mainloop.h"

MainLoop::MainLoop(QJsonObject const &a_config)
{
  m_timer = new QTimer(this);
  m_timer->start(1500);

  m_tensor = new TensorGraph(a_config[TENSORFLOW].toObject());
  // m_tensor->moveToThread(&m_workerTensor);
  // connect(&m_workerTensor, &QThread::finished, m_tensor, &QObject::deleteLater);
  // m_workerTensor.start();
  // connect(m_timer, &QTimer::timeout, m_tensor, &TensorGraph::onUpdate);

  cv::Mat image{ cv::imread("1.png") };

  m_tensor->selectImage(image);

  struct allScoreData allScoreDataOutput;

  allScoreDataOutput = m_tensor->returnAllLabel();

  for (unsigned int pos = 0; pos < allScoreDataOutput.labels.size(); pos++)
  {
    H_Logger->trace("[{}]:{}", pos, static_cast<double>(allScoreDataOutput.scores[pos]));
  }

  struct topScore topScoreOutput;

  topScoreOutput = m_tensor->returnTopLabel();

  H_Logger->trace("topScoreOutput:{}:{}", topScoreOutput.label, topScoreOutput.score);
}

MainLoop::~MainLoop() {}
