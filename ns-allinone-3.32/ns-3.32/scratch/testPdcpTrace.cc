/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012-2018 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/config-store-module.h"
#include "ns3/lte-pdcp.h"
#include <vector>
#include "ns3/netanim-module.h"
using namespace ns3;

NS_LOG_COMPONENT_DEFINE("LenaX2HandoverExample");

void NotifyConnectionEstablishedUe(std::string context,
                                   uint64_t imsi,
                                   uint16_t cellid,
                                   uint16_t rnti)
{
    // std::cout << Simulator::Now().GetSeconds() << " " << context
    //           << " UE IMSI " << imsi
    //           << ": connected to CellId " << cellid
    //           << " with RNTI " << rnti
    //           << std::endl;
}

void NotifyHandoverStartUe(std::string context,
                           uint64_t imsi,
                           uint16_t cellid,
                           uint16_t rnti,
                           uint16_t targetCellId)
{
    // std::cout << Simulator::Now().GetSeconds() << " " << context
    //           << " UE IMSI " << imsi
    //           << ": previously connected to CellId " << cellid
    //           << " with RNTI " << rnti
    //           << ", doing handover to CellId " << targetCellId
    //           << std::endl;
}

void NotifyHandoverEndOkUe(std::string context,
                           uint64_t imsi,
                           uint16_t cellid,
                           uint16_t rnti)
{
    // std::cout << Simulator::Now().GetSeconds() << " " << context
    //           << " UE IMSI " << imsi
    //           << ": successful handover to CellId " << cellid
    //           << " with RNTI " << rnti
    //           << std::endl;
}

void NotifyConnectionEstablishedEnb(std::string context,
                                    uint64_t imsi,
                                    uint16_t cellid,
                                    uint16_t rnti)
{
    // std::cout << Simulator::Now().GetSeconds() << " " << context
    //           << " eNB CellId " << cellid
    //           << ": successful connection of UE with IMSI " << imsi
    //           << " RNTI " << rnti
    //           << std::endl;
}

void NotifyHandoverStartEnb(std::string context,
                            uint64_t imsi,
                            uint16_t cellid,
                            uint16_t rnti,
                            uint16_t targetCellId)
{
    // std::cout << Simulator::Now().GetSeconds() << " " << context
    //           << " eNB CellId " << cellid
    //           << ": start handover of UE with IMSI " << imsi
    //           << " RNTI " << rnti
    //           << " to CellId " << targetCellId
    //           << std::endl;
}

void NotifyHandoverEndOkEnb(std::string context,
                            uint64_t imsi,
                            uint16_t cellid,
                            uint16_t rnti)
{
    // std::cout << Simulator::Now().GetSeconds() << " " << context
    //           << " eNB CellId " << cellid
    //           << ": completed handover of UE with IMSI " << imsi
    //           << " RNTI " << rnti
    //           << std::endl;
}
uint32_t
ConvertContextToNodeId(std::string context)
{
    std::string sub = context.substr(10);
    uint32_t pos = sub.find("/Device");
    uint32_t nodeId = atoi(sub.substr(0, pos).c_str());
    return nodeId;
}
/*
void LteUePhyReportUeMeasurementsCb(std::string context, uint16_t rnti, uint16_t cellId, double rsrp, double rsrq, bool isServingCell, uint8_t componentCarrierId)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    if (std::isnan(rsrp) != true and std::isnan(rsrq) != true)
    {
        std::cout << "nodeId" << nodeId << "cellId" << cellId << "rsrq" << rsrq << std::endl;
    }
}

*/
std::map<uint16_t, uint64_t> tmpDelay;
void delayPerSecond()
{
    Simulator::Schedule(MilliSeconds(1000), &delayPerSecond);
    std::cout << Simulator::Now().GetSeconds() << std::endl;
    for (const auto &delay : tmpDelay)
    {
        std::cout << "NodeId : " << delay.first << "delay : " << (delay.second * 10e-6) << "ms" << std::endl;
    }
    // fgetc(stdin);
}
void TestRxPDU(std::string context,
               uint16_t rnti,
               u_char lcid,
               uint32_t size,
               ulong delay)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    // std::cout << Simulator::Now().GetSeconds() << " NodeId: " << nodeId << " "
    //           << "delay: " << delay * 10e-6 << " ms" << std::endl;
    tmpDelay[nodeId] = delay;
    // std::cout << Simulator::Now().GetSeconds() << " " << context
    //           << " RNTI " << rnti
    //           << " LCID" << (unsigned)lcid
    //           << " packet size" << size
    //           << " packet delay " << delay
    //           << std::endl;
    // tmpDelay.at(rnti)
}
void pdcpDelay()
{
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/LteUeRrc/DataRadioBearerMap/*/LtePdcp/RxPDU",
                    MakeCallback(&TestRxPDU));
    // Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/LteUeRrc/Srb0/LtePdcp/RxPDU",
    //                         MakeCallback(&TestRxPDU));
    // Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/LteUeRrc/Srb1/LtePdcp/RxPDU",
    // MakeCallback(&TestRxPDU));

    // Config::ConnectFailSafe("NodeList/*/DeviceList/*/$ns3::LteNetDevice/$ns3::LteUeNetDevice/LteUeRrc/Srb0/LtePdcp/RxPDU",
    //                         MakeCallback(&TestRxPDU));

    // Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::LteNetDevice/$ns3::LteUeNetDevice/LteUeRrc/Srb1/LtePdcp/RxPDU",
    //                         MakeCallback(&TestRxPDU));

    // Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::LteNetDevice/$ns3::LteEnbNetDevice/LteEnbRrc/UeMap/*/DataRadioBearerMap/*/LtePdcp/RxPDU",
    //                         MakeCallback(&TestRxPDU));
    // Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::LteNetDevice/$ns3::LteEnbNetDevice/LteEnbRrc/UeMap/*/Srb0/LtePdcp/RxPDU",
    //                         MakeCallback(&TestRxPDU));

    // Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::LteNetDevice/$ns3::LteEnbNetDevice/LteEnbRrc/UeMap/*/Srb1/LtePdcp/RxPDU",
    //                         MakeCallback(&TestRxPDU));

    // Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::LteEnbNetDevice/LteEnbRrc/UeMap/*/DataRadioBearerMap/*/LtePdcp/RxPDU",
    //                         MakeCallback(&TestRxPDU));

    // Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::LteEnbNetDevice/LteEnbRrc/UeMap/*/Srb0/LtePdcp/RxPDU",
    //                         MakeCallback(&TestRxPDU));

    // Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::LteEnbNetDevice/LteEnbRrc/UeMap/*/Srb1/LtePdcp/ RxPDU ",
    //                         MakeCallback(&TestRxPDU));
}
/**
 * Sample simulation script for a X2-based handover.
 * It instantiates two eNodeB, attaches one UE to the 'source' eNB and
 * triggers a handover of the UE towards the 'target' eNB.
 */
int main(int argc, char *argv[])
{
    uint16_t numberOfUes = 10;
    uint16_t numberOfEnbs = 1;
    uint16_t numBearersPerUe = 1;
    Time simTime = MilliSeconds(5000);
    // double distance = 100.0;
    bool disableDl = false;
    bool disableUl = false;

    // change some default attributes so that they are reasonable for
    // this scenario, but do this before processing command line
    // arguments, so that the user is allowed to override these settings
    Config::SetDefault("ns3::RadioBearerStatsCalculator::EpochDuration", TimeValue(MilliSeconds(1000)));
    Config::SetDefault("ns3::UdpClient::Interval", TimeValue(MilliSeconds(25)));
    Config::SetDefault("ns3::UdpClient::MaxPackets", UintegerValue(1000000));
    Config::SetDefault("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue(320));
    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(false));

    // Command line arguments
    CommandLine cmd(__FILE__);
    cmd.AddValue("numberOfUes", "Number of UEs", numberOfUes);
    cmd.AddValue("numberOfEnbs", "Number of eNodeBs", numberOfEnbs);
    cmd.AddValue("simTime", "Total duration of the simulation", simTime);
    cmd.AddValue("disableDl", "Disable downlink data flows", disableDl);
    cmd.AddValue("disableUl", "Disable uplink data flows", disableUl);
    cmd.Parse(argc, argv);

    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);
    lteHelper->SetSchedulerType("ns3::RrFfMacScheduler");
    lteHelper->SetHandoverAlgorithmType("ns3::NoOpHandoverAlgorithm"); // disable automatic handover

    Ptr<Node> pgw = epcHelper->GetPgwNode();

    // Create a single RemoteHost
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    // Create the Internet
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("1Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.010)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress(1);

    // Routing of the Internet Host (towards the LTE network)
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    // interface 0 is localhost, 1 is the p2p device
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    NodeContainer ueNodes;
    NodeContainer enbNodes;
    enbNodes.Create(numberOfEnbs);
    ueNodes.Create(numberOfUes);
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        uint32_t ueNodeId = ueNodes.Get(i)->GetId();
        tmpDelay[ueNodeId] = 0;
    }

    // EpsBearer bearer = EpsBearer::NGBR_V2X;
    double requestedDelay = 0.01;
    std::cout << requestedDelay << std::endl;
    // Install Mobility Model
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    Ptr<ListPositionAllocator> positionAllocUe = CreateObject<ListPositionAllocator>();

    positionAlloc->Add(Vector(1800, 1800, 0));
    // for (uint16_t i = 0; i < numberOfEnbs; i++)
    // {
    //     positionAlloc->Add(Vector(distance * 2 * i - distance, 0, 0));
    // }
    for (uint16_t i = 0; i < numberOfUes; i++)
    {
        positionAllocUe->Add(Vector(1800, 1800, 0));
    }
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.SetPositionAllocator(positionAlloc);
    mobility.Install(enbNodes);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.SetPositionAllocator(positionAllocUe);
    mobility.Install(ueNodes);
    // mobility.SetMobilityModel("ns3::RandomDirection2dMobilityModel",
    //                           "Bounds", RectangleValue(Rectangle(0, 5000, 0, 5000)),
    //                           "Speed", StringValue("ns3::ConstantRandomVariable[Constant=1000]"),
    //                           "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0.01]"));
    // mobility.Install(ueNodes);

    // Install LTE Devices in eNB and UEs
    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);
    for (uint32_t i = 0; i < numberOfEnbs; ++i)
    {
        Ptr<LteEnbNetDevice> device = enbLteDevs.Get(i)->GetObject<LteEnbNetDevice>();
        Ptr<LteEnbPhy> phy = device->GetPhy();
        phy->SetTxPower(5); // TR 36.873
    }

    // Install the IP stack on the UEs
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIfaces;
    ueIpIfaces = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs));

    // Attach all UEs to the first eNodeB
    for (uint16_t i = 0; i < numberOfUes; i++)
    {
        lteHelper->Attach(ueLteDevs.Get(i), enbLteDevs.Get(0));
    }

    NS_LOG_LOGIC("setting up applications");

    // Install and start applications on UEs and remote host
    uint16_t dlPort = 10000;
    uint16_t ulPort = 20000;

    // randomize a bit start times to avoid simulation artifacts
    // (e.g., buffer overflows due to packet transmissions happening
    // exactly at the same time)
    Ptr<UniformRandomVariable> startTimeSeconds = CreateObject<UniformRandomVariable>();
    startTimeSeconds->SetAttribute("Min", DoubleValue(0.05));
    startTimeSeconds->SetAttribute("Max", DoubleValue(0.99));
    for (uint32_t u = 0; u < numberOfUes; ++u)
    {
        Ptr<Node> ue = ueNodes.Get(u);
        // Set the default gateway for the UE
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting(ue->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);

        for (uint32_t b = 0; b < numBearersPerUe; ++b)
        {
            ApplicationContainer clientApps;
            ApplicationContainer serverApps;
            Ptr<EpcTft> tft = Create<EpcTft>();

            if (!disableDl)
            {
                ++dlPort;

                NS_LOG_LOGIC("installing UDP DL app for UE " << u);
                UdpClientHelper dlClientHelper(ueIpIfaces.GetAddress(u), dlPort);
                clientApps.Add(dlClientHelper.Install(remoteHost));
                PacketSinkHelper dlPacketSinkHelper("ns3::UdpSocketFactory",
                                                    InetSocketAddress(Ipv4Address::GetAny(), dlPort));
                serverApps.Add(dlPacketSinkHelper.Install(ue));

                EpcTft::PacketFilter dlpf;
                dlpf.localPortStart = dlPort;
                dlpf.localPortEnd = dlPort;
                tft->Add(dlpf);
            }

            if (!disableUl)
            {
                ++ulPort;

                NS_LOG_LOGIC("installing UDP UL app for UE " << u);
                UdpClientHelper ulClientHelper(remoteHostAddr, ulPort);
                clientApps.Add(ulClientHelper.Install(ue));
                PacketSinkHelper ulPacketSinkHelper("ns3::UdpSocketFactory",
                                                    InetSocketAddress(Ipv4Address::GetAny(), ulPort));
                serverApps.Add(ulPacketSinkHelper.Install(remoteHost));

                EpcTft::PacketFilter ulpf;
                ulpf.remotePortStart = ulPort;
                ulpf.remotePortEnd = ulPort;
                tft->Add(ulpf);
            }

            EpsBearer bearer(EpsBearer::NGBR_VIDEO_TCP_DEFAULT);
            lteHelper->ActivateDedicatedEpsBearer(ueLteDevs.Get(u), bearer, tft);

            Time startTime = Seconds(startTimeSeconds->GetValue());
            serverApps.Start(startTime);
            clientApps.Start(startTime);
            clientApps.Stop(simTime);

        } // end for b
    }

    // Uncomment to enable PCAP tracing
    // p2ph.EnablePcapAll("lena-x2-handover");

    // lteHelper->EnablePhyTraces();
    // lteHelper->EnableMacTraces();
    lteHelper->EnableRlcTraces();
    lteHelper->EnablePdcpTraces();
    Ptr<RadioBearerStatsCalculator> rlcStats = lteHelper->GetRlcStats();
    rlcStats->SetAttribute("EpochDuration", TimeValue(Seconds(0.05)));
    Ptr<RadioBearerStatsCalculator> pdcpStats = lteHelper->GetPdcpStats();
    pdcpStats->SetAttribute("EpochDuration", TimeValue(Seconds(1)));

    // connect custom trace sinks for RRC connection establishment and handover notification
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteEnbNetDevice/LteEnbRrc/ConnectionEstablished",
                    MakeCallback(&NotifyConnectionEstablishedEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
                    MakeCallback(&NotifyConnectionEstablishedUe));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
                    MakeCallback(&NotifyHandoverStartEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/HandoverStart",
                    MakeCallback(&NotifyHandoverStartUe));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
                    MakeCallback(&NotifyHandoverEndOkEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/HandoverEndOk",
                    MakeCallback(&NotifyHandoverEndOkUe));
//    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/ComponentCarrierMapUe/*/LteUePhy/ReportUeMeasurements", MakeCallback(&LteUePhyReportUeMeasurementsCb));

    Simulator::Schedule(MilliSeconds(1000), &pdcpDelay);
    Simulator::Schedule(MilliSeconds(1000), &delayPerSecond);
    Simulator::Stop(simTime + MilliSeconds(200));
    Simulator::Run();

    // GtkConfigStore config;
    // config.ConfigureAttributes ();

    Simulator::Destroy();
    return 0;
}
