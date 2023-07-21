#include <numeric>
#include <cmath>
#include <regex>
#include "ns3/core-module.h"
#include "ns3/lte-module.h"
#include "ns3/node-container.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/opengym-module.h"

using namespace ns3;
int main(int argc, char *argv[])
{
  std::string mobility_file = "/home/steven/ns-allinone-3.32/ns-3.32/Data/env_sumoTest/ns2mobility.tcl";
  uint32_t nUeNodes = 1;
  NodeContainer ueNodes;
  ueNodes.Create(nUeNodes);
  Ns2MobilityHelper sumo_Trace(mobility_file);
  sumo_Trace.Install();
  // animation
  AnimationInterface anim("test.xml");
  for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
  {
    anim.UpdateNodeColor(ueNodes.Get(i), 0, 0, 255);
    anim.UpdateNodeSize(ueNodes.Get(i)->GetId(), 5.0, 5.0);
  }

  Time simStopTime = Seconds(210);

  Simulator::Stop(simStopTime);
  Simulator::Run();

  Simulator::Destroy();

  return 0;
}