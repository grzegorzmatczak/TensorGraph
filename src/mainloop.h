#ifndef MAINLOOP_H
#define MAINLOOP_H

#include <QJsonObject>
#include <QTimer>
#include <QObject>
#include <QThread>

#include "tensorgraph.h"

class MainLoop : public QObject
{
  Q_OBJECT
 public:
  MainLoop(QJsonObject const &a_config);
  ~MainLoop();
 signals:

 public slots:

private:
  QTimer *m_timer;

  QThread m_workerTensor;
  TensorGraph *m_tensor;

  QString m_infoTopic;
  QString m_infoTopicClient;
};

#endif // MAINLOOP_H
