#include <QApplication>

#include "../../ConfigReader/src/configreader.h"
#include "../../IncludeSpdlog/spdlog.h"

#include "mainloop.h"
#include "structures.h"

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);

  H_Logger->set_level(static_cast<spdlog::level::level_enum>(1));

  QString configName{ "config.json" };
  ConfigReader *_configReader = new ConfigReader();
  QString possibleError;
  QJsonObject jObject;
  if (!_configReader->readConfig(configName, jObject, possibleError))
  {
    H_Logger->error("{}", possibleError.toStdString());
    return -1;
  }
  delete _configReader;

  QJsonObject jDataset = jObject[DATASET].toObject();
  qint32 messageLevel{ jObject[GENERAL].toObject()[LOG_LEVEL].toInt() };
  H_Logger->info("messageLevel:{}", messageLevel);
  H_Logger->set_level(static_cast<spdlog::level::level_enum>(messageLevel));
  H_Logger->set_pattern("[%Y-%m-%d] [%H:%M:%S.%e] [%t] [%^%l%$] %v");

  MainLoop loop{ jObject };
  return app.exec();
}
