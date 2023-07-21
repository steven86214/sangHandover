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
#include <ns3/buildings-module.h>
#include "ns3/lte-pdcp.h"
using namespace ns3;

// types
struct QosRequirement
{
    enum EpsBearer::Qci qci;
    uint64_t bitrate; // unit: bps
};

struct UeQoeFlowState
{
    double qoeScore;
    double achievedBitrate;
    double achievedDelay;
    double achievedErrorRate;
};

struct UeQosRequirementState
{
    bool isNgbr;
    double requestedBitrate;
    double requestedDelay;
    double requestedErrorRate;
};

struct CellIdRnti
{
    uint16_t cellId; //!< Cell Id
    uint16_t rnti;   //!< RNTI
};

struct BoundCallbackArgument : public SimpleRefCount<BoundCallbackArgument>
{
public:
    uint64_t imsi;   //!< imsi
    uint16_t cellId; //!< cellId
};

// static global variables
static const std::vector<QosRequirement> qosRequirementOptions = {
    {EpsBearer::GBR_CONV_VOICE, 100000}, // 0
    //
    {EpsBearer::GBR_CONV_VIDEO, 300000},  // 1
    {EpsBearer::GBR_CONV_VIDEO, 500000},  // 2
    {EpsBearer::GBR_CONV_VIDEO, 1500000}, // 3
    {EpsBearer::GBR_CONV_VIDEO, 2000000}, // 4
    {EpsBearer::GBR_CONV_VIDEO, 4000000}, // 5
    {EpsBearer::GBR_CONV_VIDEO, 8000000}, // 6
    //
    {EpsBearer::GBR_GAMING, 64000}, // 7
    //
    {EpsBearer::GBR_NON_CONV_VIDEO, 1000000},  // 8
    {EpsBearer::GBR_NON_CONV_VIDEO, 2500000},  // 9
    {EpsBearer::GBR_NON_CONV_VIDEO, 5000000},  // 10
    {EpsBearer::GBR_NON_CONV_VIDEO, 8000000},  // 11
    {EpsBearer::GBR_NON_CONV_VIDEO, 16000000}, // 12
    {EpsBearer::GBR_NON_CONV_VIDEO, 45000000}, // 13
    //
    {EpsBearer::GBR_MC_PUSH_TO_TALK, 100000}, // 14
    //
    {EpsBearer::GBR_NMC_PUSH_TO_TALK, 100000}, // 15
    //
    {EpsBearer::GBR_MC_VIDEO, 300000},  // 16
    {EpsBearer::GBR_MC_VIDEO, 500000},  // 17
    {EpsBearer::GBR_MC_VIDEO, 1500000}, // 18
    {EpsBearer::GBR_MC_VIDEO, 2000000}, // 19
    {EpsBearer::GBR_MC_VIDEO, 4000000}, // 20
    {EpsBearer::GBR_MC_VIDEO, 8000000}, // 21
    //
    {EpsBearer::NGBR_IMS, 100000}, // 22
    //
    {EpsBearer::NGBR_VIDEO_TCP_OPERATOR, 1000000},  // 23
    {EpsBearer::NGBR_VIDEO_TCP_OPERATOR, 2500000},  // 24
    {EpsBearer::NGBR_VIDEO_TCP_OPERATOR, 5000000},  // 25
    {EpsBearer::NGBR_VIDEO_TCP_OPERATOR, 8000000},  // 26
    {EpsBearer::NGBR_VIDEO_TCP_OPERATOR, 16000000}, // 27
    {EpsBearer::NGBR_VIDEO_TCP_OPERATOR, 45000000}, // 28
    //
    {EpsBearer::NGBR_VOICE_VIDEO_GAMING, 300000},  // 29
    {EpsBearer::NGBR_VOICE_VIDEO_GAMING, 500000},  // 30
    {EpsBearer::NGBR_VOICE_VIDEO_GAMING, 1500000}, // 31
    {EpsBearer::NGBR_VOICE_VIDEO_GAMING, 2000000}, // 32
    {EpsBearer::NGBR_VOICE_VIDEO_GAMING, 4000000}, // 33
    {EpsBearer::NGBR_VOICE_VIDEO_GAMING, 8000000}, // 34
    //
    {EpsBearer::NGBR_VIDEO_TCP_PREMIUM, 1000000},  // 35
    {EpsBearer::NGBR_VIDEO_TCP_PREMIUM, 2500000},  // 36
    {EpsBearer::NGBR_VIDEO_TCP_PREMIUM, 5000000},  // 37
    {EpsBearer::NGBR_VIDEO_TCP_PREMIUM, 8000000},  // 38
    {EpsBearer::NGBR_VIDEO_TCP_PREMIUM, 16000000}, // 39
    {EpsBearer::NGBR_VIDEO_TCP_PREMIUM, 45000000}, // 40
    //
    {EpsBearer::NGBR_VIDEO_TCP_DEFAULT, 1000000},  // 41
    {EpsBearer::NGBR_VIDEO_TCP_DEFAULT, 2500000},  // 42
    {EpsBearer::NGBR_VIDEO_TCP_DEFAULT, 5000000},  // 43
    {EpsBearer::NGBR_VIDEO_TCP_DEFAULT, 8000000},  // 44
    {EpsBearer::NGBR_VIDEO_TCP_DEFAULT, 16000000}, // 45
    {EpsBearer::NGBR_VIDEO_TCP_DEFAULT, 45000000}, // 46
    //
    {EpsBearer::NGBR_MC_DELAY_SIGNAL, 100000}, // 47
    //
    {EpsBearer::NGBR_MC_DATA, 1000000},  // 48
    {EpsBearer::NGBR_MC_DATA, 2500000},  // 49
    {EpsBearer::NGBR_MC_DATA, 5000000},  // 50
    {EpsBearer::NGBR_MC_DATA, 8000000},  // 51
    {EpsBearer::NGBR_MC_DATA, 16000000}, // 52
    {EpsBearer::NGBR_MC_DATA, 45000000}, // 53
    //
    {EpsBearer::NGBR_V2X, 25000000},   // 54
    {EpsBearer::NGBR_V2X, 50000000},   // 55
    {EpsBearer::NGBR_V2X, 65000000},   // 56
    {EpsBearer::NGBR_V2X, 1000000000}, // 57
    //
    {EpsBearer::NGBR_LOW_LAT_EMBB, 25000000},  // 58
    {EpsBearer::NGBR_LOW_LAT_EMBB, 100000000}, // 59
    //
    {EpsBearer::DGBR_DISCRETE_AUT_SMALL, 200000}, // 60
    //
    {EpsBearer::DGBR_DISCRETE_AUT_LARGE, 1000000}, // 61
    //
    {EpsBearer::DGBR_ITS, 350000}, // 62
    //
    {EpsBearer::DGBR_ELECTRICITY, 400000}, // 63
    //
    {EpsBearer::GBR_CONV_VOICE, 160000}, // 64
};

static const std::vector<std::string> schedulerNameOptions = {
    "ns3::PfFfMacScheduler",
    "ns3::PssFfMacScheduler",
    "ns3::CqaFfMacScheduler",
    "ns3::FdBetFfMacScheduler",
};

static const std::vector<enum EpsBearer::Qci> ngbrQciOptions = {
    EpsBearer::NGBR_IMS,
    EpsBearer::NGBR_VIDEO_TCP_OPERATOR,
    EpsBearer::NGBR_VOICE_VIDEO_GAMING,
    EpsBearer::NGBR_VIDEO_TCP_PREMIUM,
    EpsBearer::NGBR_VIDEO_TCP_DEFAULT,
    EpsBearer::NGBR_MC_DELAY_SIGNAL,
    EpsBearer::NGBR_MC_DATA,
    EpsBearer::NGBR_V2X,
    EpsBearer::NGBR_LOW_LAT_EMBB,
};

// global variables
Ptr<LteHelper> lte;
Ptr<PointToPointEpcHelper> epc;
Ptr<FlowMonitor> monitor;
Ptr<Ipv4FlowClassifier> classifier;

std::vector<uint32_t> enbNodeIds;
std::map<uint32_t, uint32_t> enbNodeIdToEnbIndexs;
std::vector<uint32_t> cellNrRbs;
std::vector<uint32_t> cellSubBandOffsets;
std::vector<uint16_t> cellIds;
std::map<uint16_t, Ptr<LteEnbNetDevice>> cellIdToPtrLteEnbNetDevices;
std::map<uint32_t, uint32_t> ipv4AddressToUeNodeIds;

std::map<uint32_t, uint16_t> ueNodeIdToServingCellIds;
std::map<uint32_t, Time> ueNodeIdToServingCellIdTimes;
std::map<uint32_t, uint16_t> ueNodeIdToPrevServingCellIds;
std::map<uint32_t, Time> ueNodeIdToPrevServingCellIdTimes;
std::map<uint32_t, uint16_t> ueNodeIdToPrevPrevServingCellIds;
std::map<uint32_t, Time> ueNodeIdToPrevPrevServingCellIdTimes;
std::map<uint32_t, uint32_t> ueNodeIdToTotalNrRlfs;
std::map<uint32_t, uint32_t> ueNodeIdToTotalNrHandovers;
std::map<uint32_t, uint32_t> ueNodeIdToTotalNrPingpongs;
std::map<uint32_t, std::map<uint16_t, double>> ueNodeIdToCellIdToRsrps;
std::map<uint32_t, std::map<uint16_t, double>> ueNodeIdToCellIdToRsrqs;
std::map<uint32_t, std::map<uint16_t, double>> ueNodeIdToCellIdToSinrs;
std::map<uint32_t, uint32_t> ueNodeIdToTotalRxBytes;
std::map<uint32_t, uint32_t> ueNodeIdToNrPhyRxEndOks;
std::map<uint32_t, uint32_t> ueNodeIdToNrPhyRxEndErrors;
std::map<uint32_t, uint32_t> enbNodeIdToNrPhyTxEnds;
std::map<uint32_t, std::deque<DlSchedulingCallbackInfo>> enbNodeIdToDlSchedInfoQues;

std::map<uint32_t, uint32_t> ueNodeIdToQosRequirementChoices;
std::map<uint16_t, double> cellIdToBwHzs;
std::map<uint32_t, double> ueNodeIdToPrevSumDelays;
std::map<uint32_t, double> ueNodeIdToPerSecSumDelays;
std::map<uint32_t, uint32_t> ueNodeIdToPrevNrRxPackets;
std::map<uint32_t, uint32_t> ueNodeIdToPerSecNrRxPackets;
std::map<uint32_t, uint64_t> ueNodeIdToPrevSumRxBytes;
std::map<uint32_t, uint32_t> ueNodeIdToPerSecSumRxBytes;
std::map<uint32_t, uint32_t> ueNodeIdToPrevNrTxPackets;
std::map<uint32_t, uint32_t> ueNodeIdToPerSecNrTxPackets;
// std::map<uint32_t, uint32_t> ueNodeIdToPrevNrLostPackets;
// std::map<uint32_t, uint32_t> ueNodeIdToPerSecNrLostPackets;
std::map<uint32_t, uint32_t> ueNodeIdToNrRxEndOks;
std::map<uint32_t, uint32_t> ueNodeIdToPrevNrRxEndOks;
std::map<uint32_t, uint32_t> ueNodeIdToPerSecNrRxEndOks;
std::map<uint32_t, uint32_t> ueNodeIdToNrRxEndErrors;
std::map<uint32_t, uint32_t> ueNodeIdToPrevNrRxEndErrors;
std::map<uint32_t, uint32_t> ueNodeIdToPerSecNrRxEndErrors;

std::map<uint32_t, UeQoeFlowState> ueNodeIdToUeQoeFlowStates;

uint32_t currentHandoverUeIndex;

std::map<uint32_t, double> ueNodeIdToTriggerPrevSumDelays;
std::map<uint32_t, double> ueNodeIdToTriggerOneSecSumDelays;
std::map<uint32_t, uint32_t> ueNodeIdToTriggerPrevNrRxPackets;
std::map<uint32_t, uint32_t> ueNodeIdToTriggerOneSecNrRxPackets;
std::map<uint32_t, uint64_t> ueNodeIdToTriggerPrevSumRxBytes;
std::map<uint32_t, uint32_t> ueNodeIdToTriggerOneSecSumRxBytes;
std::map<uint32_t, uint32_t> ueNodeIdToTriggerPrevNrTxPackets;
std::map<uint32_t, uint32_t> ueNodeIdToTriggerOneSecNrTxPackets;
std::map<uint32_t, uint32_t> ueNodeIdToTriggerPrevNrRxEndOks;
std::map<uint32_t, uint32_t> ueNodeIdToTriggerOneSecNrRxEndOks;
std::map<uint32_t, uint32_t> ueNodeIdToTriggerPrevNrRxEndErrors;
std::map<uint32_t, uint32_t> ueNodeIdToTriggerOneSecNrRxEndErrors;

std::map<uint32_t, uint32_t> nrPacketReceive;
std::map<uint32_t, uint32_t> nrPacketSend;
std::map<uint32_t, uint32_t> nrPrevPacketReceive;
std::map<uint32_t, uint32_t> nrPrevPacketSend;

std::map<uint32_t, uint32_t> enbNodeIdToNrUsedSubframes;

std::map<uint32_t, uint32_t> enbNodeIdToNrDlPhyTxs;
std::map<uint32_t, uint32_t> enbNodeIdToPrevNrDlPhyTxs;
std::map<uint32_t, uint32_t> enbNodeIdToPerSecNrDlPhyTxs;
std::map<uint32_t, uint32_t> enbNodeIdToNrDlPhyNdis;
std::map<uint32_t, uint32_t> enbNodeIdToPrevNrDlPhyNdis;
std::map<uint32_t, uint32_t> enbNodeIdToPerSecNrDlPhyNdis;

std::map<uint32_t, uint32_t> enbNodeIdToPerSecDlDelay;
std::map<uint32_t, uint32_t> enbNodeIdToPerSecDlOutrageRatio;
std::map<uint32_t, uint32_t> ueNodeIdToPerSecDlOutrageTimeSlot;
std::map<uint32_t, uint32_t> ueNodeIdToPerSecDlTotalTimeSlot;

std::map<uint32_t, std::deque<std::pair<uint16_t, uint16_t>>> enbNodeIdToRntiMcsPairQues;
std::map<uint32_t, std::vector<uint32_t>> enbNodeIdToMcsDevDistrs;

std::map<CellIdRnti, std::string> ueManagerPathByCellIdRnti;
std::map<uint64_t, uint32_t> ueImsiToDlTxPackets;
std::map<uint64_t, uint32_t> ueImsiToDlRxPackets;
std::map<uint64_t, uint32_t> ueImsiToPrevDlTxPackets;
std::map<uint64_t, uint32_t> ueImsiToPrevDlRxPackets;
std::map<uint32_t, double> ueNodeIdToPacketErrorRates;

// std::map<uint32_t, std::deque<std::pair<uint32_t, uint32_t>>> ueImsiToEnbNodeIdMcsPairQues;
std::map<uint32_t, double> ueImsiToMeanMcss;
std::map<uint32_t, std::deque<std::pair<uint32_t, uint32_t>>> cellIdToRntiMcsPairQues;
std::map<uint32_t, std::map<uint32_t, double>> cellIdToRntiToMeanMcss;

std::map<std::string, std::vector<double>> obsUeInfo;
std::map<uint32_t, uint64_t> ueNodeIdToDelayPerSecond;
// function
std::map<std::string, std::vector<double>>
GetInfoFromTsv(std::string infoDir, std::string fn)
{
    std::map<std::string, std::vector<double>> info;
    // std::string path = "/home/mdclab/Documents/Data/env4_info/" + fn + ".tsv";
    std::string path = infoDir + fn + ".tsv";
    std::ifstream infoFile(path);
    std::string line;
    // read column name
    std::getline(infoFile, line);
    std::vector<std::string> colNames;
    std::string colName;
    std::stringstream colNameStream(line);
    while (colNameStream >> colName)
    {

        std::cout << "------------ colName " << std::endl
                  << colName << std::endl;
        colNames.push_back(colName);
        info[colName] = std::vector<double>();
    }
    // read data
    while (std::getline(infoFile, line))
    {
        uint32_t colIndex = 0;
        std::stringstream dataStream(line);
        double value;
        while (dataStream >> value)
        {
            std::string colName = colNames[colIndex];
            info[colName].push_back(value);
            ++colIndex;
        }
    }
    return info;
}

std::map<std::string, std::vector<double>>
GetEnbInfo(std::string infoDir, std::string fn)
{
    std::map<std::string, std::vector<double>> enbInfo = GetInfoFromTsv(infoDir, fn);
    return enbInfo;
}

std::map<std::string, std::vector<double>>
GetUeInfo(std::string infoDir, std::string fn)
{
    std::map<std::string, std::vector<double>> ueInfo = GetInfoFromTsv(infoDir, fn);
    return ueInfo;
}

void SetEnbNodeIds(NodeContainer enbs)
{
    enbNodeIds.clear();
    enbNodeIdToEnbIndexs.clear();
    for (uint32_t i = 0; i < enbs.GetN(); ++i)
    {
        uint32_t enbNodeId = enbs.Get(i)->GetId();
        enbNodeIds.push_back(enbNodeId);
        enbNodeIdToEnbIndexs[enbNodeId] = i;
    }
}

void SetNodeConstantPositionMobilityModel(Ptr<Node> node, Vector pos)
{
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> nodePosAlloc = CreateObject<ListPositionAllocator>();
    nodePosAlloc->Add(pos);
    mobility.SetPositionAllocator(nodePosAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(node);
}

void SetRemoteHostMobilityModel(Ptr<Node> remoteHost, Vector remoteHostPos)
{
    SetNodeConstantPositionMobilityModel(remoteHost, remoteHostPos);
}

void SetPgwMobilityModel(Ptr<Node> pgw, Vector pgwPos)
{
    SetNodeConstantPositionMobilityModel(pgw, pgwPos);
}

std::vector<Vector>
GetNodePossFromInfo(const std::map<std::string, std::vector<double>> &info)
{
    const std::vector<double> &xs = info.at("x");
    const std::vector<double> &ys = info.at("y");
    const std::vector<double> &zs = info.at("z");
    std::vector<Vector> poss;
    std::cout << "xs.size" << xs.size() << std::endl;
    for (uint32_t i = 0; i < xs.size(); ++i)
    {
        std::cout << "xs[i], ys[i], zs[i] " << xs[i] << ys[i] << zs[i] << std::endl;
        poss.push_back(Vector(xs[i], ys[i], zs[i]));
    }
    return poss;
}

std::vector<Vector>
GetEnbPoss(const std::map<std::string, std::vector<double>> &enbInfo)
{
    std::vector<Vector> poss = GetNodePossFromInfo(enbInfo);
    return poss;
}

void SetNodesConstantPositionMobilityModel(NodeContainer nodes, const std::vector<Vector> &poss)
{
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> nodePosAlloc = CreateObject<ListPositionAllocator>();
    for (const auto &pos : poss)
    {
        nodePosAlloc->Add(pos);
        // std::cout << pos.x << "  " << pos.y << "  " << pos.z << "  " << std::endl;
    }
    mobility.SetPositionAllocator(nodePosAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);
    BuildingsHelper::Install(nodes);
}

void SetEnbsMobilityModel(NodeContainer enbs, const std::vector<Vector> &enbPoss)
{
    SetNodesConstantPositionMobilityModel(enbs, enbPoss);
}

std::vector<Vector>
GetUePoss(const std::map<std::string, std::vector<double>> &uePosInfo)
{
    std::vector<Vector> poss = GetNodePossFromInfo(uePosInfo);
    return poss;
}

void SetCtrlUesMobilityModel(NodeContainer ctrlUes, std::string infoDir, std::string ctrlUeTraceFn)
{
    std::map<std::string, std::vector<double>> ctrlUePosInfo = GetInfoFromTsv(infoDir, ctrlUeTraceFn);
    std::vector<Vector> ctrlUePoss = GetUePoss(ctrlUePosInfo);
    SetNodesConstantPositionMobilityModel(ctrlUes, ctrlUePoss);
}

void SetObsUesMobilityModel(NodeContainer obsUes, std::string infoDir, std::string obsUeTraceFn)
{
    if (std::regex_match(obsUeTraceFn, std::regex{"[A-Za-z0-9]*Poss[A-Za-z0-9]*"}))
    {
        std::map<std::string, std::vector<double>> obsUePosInfo = GetInfoFromTsv(infoDir, obsUeTraceFn);
        std::vector<Vector> obsUePoss = GetUePoss(obsUePosInfo);
        SetNodesConstantPositionMobilityModel(obsUes, obsUePoss);
    }
    else if (std::regex_match(obsUeTraceFn, std::regex{"[A-Za-z0-9]*Trace[A-Za-z0-9]*"}))
    {

        std::string path = infoDir + obsUeTraceFn + ".tcl";
        Ns2MobilityHelper ns2Mobility = Ns2MobilityHelper(path);
        ns2Mobility.Install(NodeList::Begin() + obsUes.Get(0)->GetId(),
                            NodeList::Begin() + obsUes.Get(0)->GetId() + obsUes.GetN());
        BuildingsHelper::Install(obsUes);
    }
}

NetDeviceContainer
CreateEnbDevices(NodeContainer enbs,
                 const std::map<std::string, std::vector<double>> &enbInfo)
{
    NetDeviceContainer enbDevices;
    const std::vector<double> &bandwidthNrRbs = enbInfo.at("bandwidthNrRb");
    const std::vector<double> &subBandOffsets = enbInfo.at("subBandOffset");
    const std::vector<double> &subBandwidthNrRbs = enbInfo.at("subBandwidthNrRb");
    const std::vector<double> &txPowerDbms = enbInfo.at("txPowerDbm");
    const std::vector<double> &schedulerChoices = enbInfo.at("schedulerChoice");

    for (uint32_t i = 0; i < enbs.GetN(); ++i)
    {
        cellNrRbs.push_back(subBandwidthNrRbs[i]);
        cellSubBandOffsets.push_back(subBandOffsets[i]);
        lte->SetEnbDeviceAttribute("DlBandwidth", UintegerValue(bandwidthNrRbs[i]));
        lte->SetEnbDeviceAttribute("UlBandwidth", UintegerValue(bandwidthNrRbs[i]));
        lte->SetHandoverAlgorithmType("ns3::NoOpHandoverAlgorithm");
        lte->SetFfrAlgorithmType("ns3::LteFrHardAlgorithm");
        lte->SetFfrAlgorithmAttribute("DlSubBandOffset", UintegerValue(subBandOffsets[i]));
        lte->SetFfrAlgorithmAttribute("DlSubBandwidth", UintegerValue(subBandwidthNrRbs[i]));
        lte->SetFfrAlgorithmAttribute("UlSubBandOffset", UintegerValue(subBandOffsets[i]));
        lte->SetFfrAlgorithmAttribute("UlSubBandwidth", UintegerValue(subBandwidthNrRbs[i]));
        uint32_t j = schedulerChoices[i];
        lte->SetSchedulerType(schedulerNameOptions[j]);
        enbDevices.Add(lte->InstallEnbDevice(enbs.Get(i)));
        Ptr<LteEnbNetDevice> device = enbDevices.Get(i)->GetObject<LteEnbNetDevice>();
        Ptr<LteEnbPhy> phy = device->GetPhy();
        phy->SetTxPower(txPowerDbms[i]); // TR 36.873

        uint16_t cellId = device->GetCellId();
        cellIds.push_back(cellId);
        cellIdToPtrLteEnbNetDevices[cellId] = device;
    }
    return enbDevices;
}

NetDeviceContainer
CreateUeDevices(NodeContainer ues)
{
    NetDeviceContainer ueDevices;
    lte->SetFfrAlgorithmType("ns3::LteFrNoOpAlgorithm");
    ueDevices = lte->InstallUeDevice(ues);
    return ueDevices;
}

Ipv4InterfaceContainer
CreateUeInternetIpv4Interfaces(NodeContainer ues)
{
    Ipv4InterfaceContainer ueInternetIpv4Interfaces;
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        Ptr<NetDevice> ueDevice = ue->GetDevice(1);
        ueInternetIpv4Interfaces.Add(epc->AssignUeIpv4Address(NetDeviceContainer(ueDevice)));
    }

    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        uint32_t ueNodeId = ues.Get(i)->GetId();
        uint32_t ip = ueInternetIpv4Interfaces.GetAddress(i).Get();
        ipv4AddressToUeNodeIds[ip] = ueNodeId;
    }
    return ueInternetIpv4Interfaces;
}

void SetUeStaticRouting(NodeContainer ues)
{
    Ipv4StaticRoutingHelper ipv4StaticRouting;
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4StaticRouting.GetStaticRouting(ue->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epc->GetUeDefaultGatewayAddress(), 1);
    }
}

std::map<std::string, std::vector<double>>
GetCtrlUeAttachInfo(std::string infoDir, std::string fn)
{
    std::map<std::string, std::vector<double>> info = GetInfoFromTsv(infoDir, fn);
    return info;
}

void ActivateEpsBearers(NodeContainer ues,
                        const std::map<std::string, std::vector<double>> &ueInfo)
{
    const std::vector<double> &ueQosChoices = ueInfo.at("qosRequirementChoice");
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        uint32_t choice = (uint32_t)ueQosChoices[i];

        Ptr<Node> ue = ues.Get(i);
        Ptr<NetDevice> device = ue->GetDevice(1);
        GbrQosInformation qos;
        qos.gbrDl = qosRequirementOptions[choice].bitrate;
        qos.gbrUl = qos.gbrDl;
        qos.mbrDl = qos.gbrDl;
        qos.mbrUl = qos.gbrUl;

        enum EpsBearer::Qci q = qosRequirementOptions[choice].qci;
        EpsBearer bearer(q, qos);
        bearer.arp.priorityLevel = 15;
        bearer.arp.preemptionCapability = true;
        bearer.arp.preemptionVulnerability = true;
        lte->ActivateDedicatedEpsBearer(device, bearer, EpcTft::Default());
    }
}

void ActivateCtrlUeEpsBearers(NodeContainer ues,
                              const std::map<std::string, std::vector<double>> &ueInfo)
{
    ActivateEpsBearers(ues, ueInfo);
}

void ActivateObsUeEpsBearers(NodeContainer ues,
                             const std::map<std::string, std::vector<double>> &ueInfo)
{
    ActivateEpsBearers(ues, ueInfo);
}

void TraceRx(uint32_t index, Ptr<const Packet> p)
{
    // std::cout << "TRx index " << index << std::endl;
    nrPacketReceive[index] += 1;
}
void TraceTx(uint32_t index, Ptr<const Packet> p)
{
    // std::cout << "TTx index " << index << std::endl;
    nrPacketSend[index] += 1;
}
void resetRxBytes(uint32_t index)
{
    nrPacketReceive[index] = 0;
}
void resetTxBytes(uint32_t index)
{
    nrPacketSend[index] = 0;
}

ApplicationContainer
CreateServerApps(Ptr<Node> remoteHost, NodeContainer ues)
{
    ApplicationContainer serverApps;
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        UdpServerHelper server(9);
        serverApps.Add(server.Install(ues.Get(i)));
    }
    for (uint32_t index = 0; index < serverApps.GetN(); ++index)
    {
        serverApps.Get(index)->TraceConnectWithoutContext("Rx", MakeBoundCallback(&TraceRx, index));
    }
    return serverApps;
}

ApplicationContainer
CreateClientApps(Ptr<Node> remoteHost, NodeContainer ues,
                 Ipv4InterfaceContainer ueInternetIpv4Interfaces,
                 const std::map<std::string, std::vector<double>> &ueInfo)
{
    const std::vector<double> &ueQosChoices = ueInfo.at("qosRequirementChoice");

    ApplicationContainer clientApps;
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        double interval = 1000.0 * 8.0 / qosRequirementOptions[ueQosChoices[i]].bitrate / 1.1;
        UdpEchoClientHelper client(ueInternetIpv4Interfaces.GetAddress(i), 9);
        client.SetAttribute("MaxPackets", UintegerValue(1e8));
        client.SetAttribute("Interval", TimeValue(Seconds(interval)));
        client.SetAttribute("PacketSize", UintegerValue(1000));
        clientApps.Add(client.Install(remoteHost));
    }
    for (uint32_t index = 0; index < clientApps.GetN(); ++index)
    {
        clientApps.Get(index)->TraceConnectWithoutContext("Tx", MakeBoundCallback(&TraceTx, index));
    }
    return clientApps;
}

uint32_t
ConvertContextToNodeId(std::string context)
{
    std::string sub = context.substr(10);
    uint32_t pos = sub.find("/Device");
    uint32_t nodeId = atoi(sub.substr(0, pos).c_str());
    return nodeId;
}

void ShiftServingCellIdAndTime(uint32_t nodeId)
{
    ueNodeIdToPrevPrevServingCellIds[nodeId] = ueNodeIdToPrevServingCellIds[nodeId];
    ueNodeIdToPrevPrevServingCellIdTimes[nodeId] = ueNodeIdToPrevServingCellIdTimes[nodeId];
    ueNodeIdToPrevServingCellIds[nodeId] = ueNodeIdToServingCellIds[nodeId];
    ueNodeIdToPrevServingCellIdTimes[nodeId] = ueNodeIdToServingCellIdTimes[nodeId];
}

void LteUeRrcConnectionEstablishedCb(std::string context, const uint64_t imsi, const uint16_t cellId, const uint16_t rnti)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    ShiftServingCellIdAndTime(nodeId);
    ueNodeIdToServingCellIds[nodeId] = cellId;
    ueNodeIdToServingCellIdTimes[nodeId] = Simulator::Now();
}

void LteUeRrcRadioLinkFailureCb(std::string context, const uint64_t imsi, const uint16_t cellId, const uint16_t rnti)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    ShiftServingCellIdAndTime(nodeId);
    ueNodeIdToServingCellIds[nodeId] = 0;
    ueNodeIdToServingCellIdTimes[nodeId] = Simulator::Now();

    ueNodeIdToTotalNrRlfs[nodeId] += 1;
}

void LteUeRrcHandoverEndOkCb(std::string context, const uint64_t imsi, const uint16_t cellId, const uint16_t rnti)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    ShiftServingCellIdAndTime(nodeId);
    ueNodeIdToServingCellIds[nodeId] = cellId;
    ueNodeIdToServingCellIdTimes[nodeId] = Simulator::Now();
    ueNodeIdToPerSecDlTotalTimeSlot[nodeId] = 0;
    ueNodeIdToPerSecDlOutrageTimeSlot[nodeId] = 0;

    ueNodeIdToTotalNrHandovers[nodeId] += 1;
    if (ueNodeIdToServingCellIds[nodeId] == ueNodeIdToPrevPrevServingCellIds[nodeId] && ueNodeIdToPrevServingCellIds[nodeId] != 0 && (ueNodeIdToServingCellIdTimes[nodeId] - ueNodeIdToPrevPrevServingCellIdTimes[nodeId]) <= Seconds(1))
    {
        ueNodeIdToTotalNrPingpongs[nodeId] += 1;
    }
    ueNodeIdToPerSecDlOutrageTimeSlot[nodeId] = 0;
    ueNodeIdToPerSecDlTotalTimeSlot[nodeId] = 0;
    std::cout << Simulator::Now().GetSeconds() << "\t"
              << nodeId << "\t"
              << "HandoverEndOk"
              << "\t"
              << std::endl;
}

double
GetSinrDbFromRsrqDb(double rsrqDb)
{
    double rsrq = std::pow(10.0, rsrqDb / 10.0);
    double sinr = 1.0 / (1.0 / (2.0 * rsrq) - 1.0);
    double sinrDb = 10.0 * std::log10(sinr);
    return sinrDb;
}

void LteUePhyReportUeMeasurementsCb(std::string context, uint16_t rnti, uint16_t cellId, double rsrp, double rsrq, bool isServingCell, uint8_t componentCarrierId)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    if (std::isnan(rsrp) != true and std::isnan(rsrq) != true)
    {
        ueNodeIdToCellIdToRsrps[nodeId][cellId] = rsrp;
        ueNodeIdToCellIdToRsrqs[nodeId][cellId] = rsrq;
        ueNodeIdToCellIdToSinrs[nodeId][cellId] = GetSinrDbFromRsrqDb(rsrq);
    }
}

void PrintLteUePhyReportUeMeasurementsCb(std::ofstream *ofs, std::string context, uint16_t rnti, uint16_t cellId, double rsrp, double rsrq, bool isServingCell, uint8_t componentCarrierId)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    (*ofs) << Simulator::Now().GetSeconds() << "\t"
           << nodeId << "\t"
           << rnti << "\t"
           << cellId << "\t"
           << rsrp << "\t"
           << rsrq << "\t"
           << isServingCell << "\t"
           << (uint16_t)componentCarrierId << "\t"
           << std::endl;
}

void UdpServerRxWithAddressesCb(std::string context, const Ptr<const Packet> packet, const Address &srcAddress, const Address &destAddress)
{
    uint32_t nodeId = ConvertContextToNodeId(context);

    ueNodeIdToTotalRxBytes[nodeId] += packet->GetSize();
}

void DlSpectrumPhyRxEndOkCb(std::string context, Ptr<const Packet> packet)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    // std::cout << Simulator::Now ().GetSeconds () << "\t"
    //           << nodeId << "\t"
    //           << "RxEndOk" << "\t"
    //           << std::endl;
    //++ueNodeIdToNrRxEndOks[nodeId];
    ++ueNodeIdToNrPhyRxEndOks[nodeId];
}

void DlSpectrumPhyRxEndErrorCb(std::string context, Ptr<const Packet> packet)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    // std::cout << Simulator::Now ().GetSeconds () << "\t"
    //           << nodeId << "\t"
    //           << "RxEndError" << "\t"
    //           << std::endl;
    //++ueNodeIdToNrRxEndErrors[nodeId];
    ++ueNodeIdToNrPhyRxEndErrors[nodeId];
}

void DlSpectrumPhyTxEndCb(std::string context, Ptr<const PacketBurst> packet)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    ++enbNodeIdToNrPhyTxEnds[nodeId];
}

void LteEnbMacDlSchedulingCb(std::string context, DlSchedulingCallbackInfo info)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    enbNodeIdToDlSchedInfoQues[nodeId].push_back(info);
}

void DlPhyTransmissionCb(std::string context, const PhyTransmissionStatParameters params)
{
    // std::cout << Simulator::Now ().GetSeconds () << "\t"
    //           << context << "\t"
    //           << params.m_cellId << "\t"
    //           << params.m_rnti << "\t"
    //           << params.m_imsi << "\t"
    //           << (uint32_t) params.m_mcs << "\t"
    //           << (uint32_t) params.m_ndi << "\t"
    //           << std::endl;
    uint32_t nodeId = ConvertContextToNodeId(context);
    ++enbNodeIdToNrDlPhyTxs[nodeId];
    if (params.m_ndi == 1)
    {
        ++enbNodeIdToNrDlPhyNdis[nodeId];
        // ueImsiToEnbNodeIdMcsPairQues[params.m_imsi].push_back (std::make_pair (nodeId, params.m_mcs));
        cellIdToRntiMcsPairQues[params.m_cellId].push_back(std::make_pair(params.m_rnti, params.m_mcs));
    }
}

// void
// RntiMcsCb (std::string context, uint16_t rnti, uint16_t mcs)
//{
//   //std::cout << Simulator::Now ().GetSeconds () << "\t"
//   //          << context << "\t"
//   //          << rnti << "\t"
//   //          << mcs << "\t"
//   //          << std::endl;
//
//   uint32_t nodeId = ConvertContextToNodeId (context);
//   enbNodeIdToRntiMcsPairQues[nodeId].push_back (std::make_pair (rnti, mcs));
// }

bool operator<(const CellIdRnti &a, const CellIdRnti &b)
{
    return ((a.cellId < b.cellId) || ((a.cellId == b.cellId) && (a.rnti < b.rnti)));
}

void DlTxPdu(uint16_t cellId, uint64_t imsi, uint16_t rnti, uint8_t lcid, uint32_t packetSize)
{
    // ImsiLcidPair_t p (imsi, lcid);
    // dlCellId[p] = cellId;
    // flowId[p] = LteFlowId_t (rnti, lcid);
    // dlTxPackets[p]++;
    // dlTxData[p] += packetSize;

    ueImsiToDlTxPackets[imsi]++;
}

void DlTxPduCallback(Ptr<BoundCallbackArgument> arg, std::string path,
                     uint16_t rnti, uint8_t lcid, uint32_t packetSize)
{
    DlTxPdu(arg->cellId, arg->imsi, rnti, lcid, packetSize);
}

void DlRxPdu(uint16_t cellId, uint64_t imsi, uint16_t rnti, uint8_t lcid, uint32_t packetSize, uint64_t delay)
{
    // ImsiLcidPair_t p (imsi, lcid);
    // dlCellId[p] = cellId;
    // dlRxPackets[p]++;
    // dlRxData[p] += packetSize;

    ueImsiToDlRxPackets[imsi]++;

    // Uint64StatsMap::iterator it = dlDelay.find (p);
    // if (it == dlDelay.end ())
    //   {
    //     dlDelay[p] = CreateObject<MinMaxAvgTotalCalculator<uint64_t> > ();
    //     dlPduSize[p] = CreateObject<MinMaxAvgTotalCalculator<uint32_t> > ();
    //   }
    // dlDelay[p]->Update (delay);
    // dlPduSize[p]->Update (packetSize);
}

void DlRxPduCallback(Ptr<BoundCallbackArgument> arg, std::string path,
                     uint16_t rnti, uint8_t lcid, uint32_t packetSize, uint64_t delay)
{
    DlRxPdu(arg->cellId, arg->imsi, rnti, lcid, packetSize, delay);
}

void ConnectTracesDrbEnb(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti, uint8_t lcid)
{
    std::string basePath;
    basePath = context.substr(0, context.rfind("/")) + "/DataRadioBearerMap/" + std::to_string(lcid - 2);
    Ptr<BoundCallbackArgument> arg = Create<BoundCallbackArgument>();
    arg->imsi = imsi;
    arg->cellId = cellId;
    Config::Connect(basePath + "/LteRlc/TxPDU",
                    MakeBoundCallback(&DlTxPduCallback, arg));
    // Config::Connect (basePath + "/LteRlc/RxPDU",
    //                  MakeBoundCallback (&UlRxPduCallback, arg));
}

void CreatedDrbEnb(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti, uint8_t lcid)
{
    ConnectTracesDrbEnb(context, imsi, cellId, rnti, lcid);
}

void StoreUeManagerPath(std::string context, uint16_t cellId, uint16_t rnti)
{
    std::string ueManagerPath;
    ueManagerPath = context.substr(0, context.rfind("/")) + "/UeMap/" + std::to_string(rnti);
    CellIdRnti key;
    key.cellId = cellId;
    key.rnti = rnti;
    ueManagerPathByCellIdRnti[key] = ueManagerPath;

    Config::Connect(ueManagerPath + "/DrbCreated",
                    MakeCallback(&CreatedDrbEnb));
}

void NotifyNewUeContextEnb(std::string context, uint16_t cellId, uint16_t rnti)
{
    StoreUeManagerPath(context, cellId, rnti);
}

void ConnectTracesSrb0(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    std::string ueRrcPath = context.substr(0, context.rfind("/"));
    CellIdRnti key;
    key.cellId = cellId;
    key.rnti = rnti;
    std::map<CellIdRnti, std::string>::iterator it = ueManagerPathByCellIdRnti.find(key);
    NS_ASSERT(it != ueManagerPathByCellIdRnti.end());
    std::string ueManagerPath = it->second;
    Ptr<BoundCallbackArgument> arg = Create<BoundCallbackArgument>();
    arg->imsi = imsi;
    arg->cellId = cellId;
    // Config::Connect (ueRrcPath + "/Srb0/LteRlc/TxPDU",
    //                  MakeBoundCallback (&UlTxPduCallback, arg));
    Config::Connect(ueRrcPath + "/Srb0/LteRlc/RxPDU",
                    MakeBoundCallback(&DlRxPduCallback, arg));
    Config::Connect(ueManagerPath + "/Srb0/LteRlc/TxPDU",
                    MakeBoundCallback(&DlTxPduCallback, arg));
    // Config::Connect (ueManagerPath + "/Srb0/LteRlc/RxPDU",
    //                  MakeBoundCallback (&UlRxPduCallback, arg));
}

void NotifyRandomAccessSuccessfulUe(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    ConnectTracesSrb0(context, imsi, cellId, rnti);
}

void ConnectTracesSrb1(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    std::string ueRrcPath = context.substr(0, context.rfind("/"));
    CellIdRnti key;
    key.cellId = cellId;
    key.rnti = rnti;
    std::map<CellIdRnti, std::string>::iterator it = ueManagerPathByCellIdRnti.find(key);
    NS_ASSERT(it != ueManagerPathByCellIdRnti.end());
    std::string ueManagerPath = it->second;
    Ptr<BoundCallbackArgument> arg = Create<BoundCallbackArgument>();
    arg->imsi = imsi;
    arg->cellId = cellId;
    // Config::Connect (ueRrcPath + "/Srb1/LteRlc/TxPDU",
    //                  MakeBoundCallback (&UlTxPduCallback, arg));
    Config::Connect(ueRrcPath + "/Srb1/LteRlc/RxPDU",
                    MakeBoundCallback(&DlRxPduCallback, arg));
    Config::Connect(ueManagerPath + "/Srb1/LteRlc/TxPDU",
                    MakeBoundCallback(&DlTxPduCallback, arg));
    // Config::Connect (ueManagerPath + "/Srb1/LteRlc/RxPDU",
    //                  MakeBoundCallback (&UlRxPduCallback, arg));
}

void CreatedSrb1Ue(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    ConnectTracesSrb1(context, imsi, cellId, rnti);
}

void ConnectTracesDrbUe(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti, uint8_t lcid)
{
    std::string basePath;
    basePath = context.substr(0, context.rfind("/")) + "/DataRadioBearerMap/" + std::to_string(lcid);
    Ptr<BoundCallbackArgument> arg = Create<BoundCallbackArgument>();
    arg->imsi = imsi;
    arg->cellId = cellId;
    // Config::Connect (basePath + "/LteRlc/TxPDU",
    //                  MakeBoundCallback (&UlTxPduCallback, arg));
    Config::Connect(basePath + "/LteRlc/RxPDU",
                    MakeBoundCallback(&DlRxPduCallback, arg));
}

void CreatedDrbUe(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti, uint8_t lcid)
{
    ConnectTracesDrbUe(context, imsi, cellId, rnti, lcid);
}

void EnsureConnected()
{
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/NewUeContext",
                    MakeCallback(&NotifyNewUeContextEnb));

    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/RandomAccessSuccessful",
                    MakeCallback(&NotifyRandomAccessSuccessfulUe));

    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/Srb1Created",
                    MakeCallback(&CreatedSrb1Ue));

    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/DrbCreated",
                    MakeCallback(&CreatedDrbUe));
}

void InitUeNodeIdToServingCellIds(NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        uint32_t ueNodeId = ues.Get(i)->GetId();
        ueNodeIdToPrevPrevServingCellIds[ueNodeId] = 0;
        ueNodeIdToPrevPrevServingCellIdTimes[ueNodeId] = Seconds(0);
        ueNodeIdToPrevServingCellIds[ueNodeId] = 0;
        ueNodeIdToPrevServingCellIdTimes[ueNodeId] = Seconds(0);
        ueNodeIdToServingCellIds[ueNodeId] = 0;
        ueNodeIdToServingCellIdTimes[ueNodeId] = Seconds(0);
        ueNodeIdToPerSecDlOutrageTimeSlot[ueNodeId] = 0;
        ueNodeIdToPerSecDlTotalTimeSlot[ueNodeId] = 0;
        ueNodeIdToDelayPerSecond[ueNodeId] = 0;
    }
}

void SetUeNodeIdToQosRequirementChoices(NodeContainer ues, const std::map<std::string, std::vector<double>> &ueInfo)
{
    const std::vector<double> &ueQosChoices = ueInfo.at("qosRequirementChoice");
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        uint32_t ueNodeId = ues.Get(i)->GetId();
        uint32_t choice = (uint32_t)ueQosChoices[i];
        ueNodeIdToQosRequirementChoices[ueNodeId] = choice;
    }
}

void SetCellIdToBwHzs(NodeContainer enbs,
                      const std::map<std::string, std::vector<double>> &enbInfo)
{
    const std::vector<double> &subBandwidthNrRbs = enbInfo.at("subBandwidthNrRb");
    for (uint32_t i = 0; i < enbs.GetN(); ++i)
    {
        uint16_t cellId = enbs.Get(i)->GetDevice(0)->GetObject<LteEnbNetDevice>()->GetCellId();
        cellIdToBwHzs[cellId] = subBandwidthNrRbs[i] * 2e5;
    }
}

void InitUeFlowStats(NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        uint32_t ueNodeId = ues.Get(i)->GetId();
        ueNodeIdToPrevSumDelays[ueNodeId] = 0.0;
        ueNodeIdToPerSecSumDelays[ueNodeId] = 0.0;
        ueNodeIdToPrevNrRxPackets[ueNodeId] = 0;
        ueNodeIdToPerSecNrRxPackets[ueNodeId] = 0;
        ueNodeIdToPrevSumRxBytes[ueNodeId] = 0;
        ueNodeIdToPerSecSumRxBytes[ueNodeId] = 0;
        ueNodeIdToPrevNrTxPackets[ueNodeId] = 0;
        ueNodeIdToPerSecNrTxPackets[ueNodeId] = 0;
        // ueNodeIdToPrevNrLostPackets[ueNodeId] = 0;
        // ueNodeIdToPerSecNrLostPackets[ueNodeId] = 0;
        ueNodeIdToNrRxEndOks[ueNodeId] = 0;
        ueNodeIdToPrevNrRxEndOks[ueNodeId] = 0;
        ueNodeIdToPerSecNrRxEndOks[ueNodeId] = 0;
        ueNodeIdToNrRxEndErrors[ueNodeId] = 0;
        ueNodeIdToPrevNrRxEndErrors[ueNodeId] = 0;
        ueNodeIdToPerSecNrRxEndErrors[ueNodeId] = 0;
        ueNodeIdToPerSecDlOutrageTimeSlot[ueNodeId] = 0;
        ueNodeIdToPerSecDlTotalTimeSlot[ueNodeId] = 0;
    }
}

void UpdateUeFlowStats()
{
    auto stats = monitor->GetFlowStats();
    for (auto i = stats.cbegin(); i != stats.cend(); ++i)
    {
        uint32_t ip = classifier->FindFlow(i->first).destinationAddress.Get();
        uint32_t ueNodeId = ipv4AddressToUeNodeIds[ip];
        ueNodeIdToPerSecSumDelays[ueNodeId] = i->second.delaySum.GetSeconds() - ueNodeIdToPrevSumDelays[ueNodeId];
        ueNodeIdToPrevSumDelays[ueNodeId] = i->second.delaySum.GetSeconds();
        ueNodeIdToPerSecNrRxPackets[ueNodeId] = i->second.rxPackets - ueNodeIdToPrevNrRxPackets[ueNodeId];
        ueNodeIdToPrevNrRxPackets[ueNodeId] = i->second.rxPackets;
        ueNodeIdToPerSecSumRxBytes[ueNodeId] = i->second.rxBytes - ueNodeIdToPrevSumRxBytes[ueNodeId];
        ueNodeIdToPrevSumRxBytes[ueNodeId] = i->second.rxBytes;
        ueNodeIdToPerSecNrTxPackets[ueNodeId] = i->second.txPackets - ueNodeIdToPrevNrTxPackets[ueNodeId];
        ueNodeIdToPrevNrTxPackets[ueNodeId] = i->second.txPackets;
        // ueNodeIdToPerSecNrLostPackets[ueNodeId] = i->second.lostPackets - ueNodeIdToPrevNrLostPackets[ueNodeId];
        // ueNodeIdToPrevNrLostPackets[ueNodeId] = i->second.lostPackets;
        ueNodeIdToPerSecNrRxEndOks[ueNodeId] = ueNodeIdToNrRxEndOks[ueNodeId] - ueNodeIdToPrevNrRxEndOks[ueNodeId];
        ueNodeIdToPrevNrRxEndOks[ueNodeId] = ueNodeIdToNrRxEndOks[ueNodeId];
        ueNodeIdToPerSecNrRxEndErrors[ueNodeId] = ueNodeIdToNrRxEndErrors[ueNodeId] - ueNodeIdToPrevNrRxEndErrors[ueNodeId];
        ueNodeIdToPrevNrRxEndErrors[ueNodeId] = ueNodeIdToNrRxEndErrors[ueNodeId];
        if (Simulator::Now().GetSeconds() > 1)
        {
            std::cout << "ueNodeId in sim.cc" << ueNodeId << std::endl;
            // double OR = ueNodeIdToPerSecDlOutrageTimeSlot[ueNodeId] / ueNodeIdToPerSecDlTotalTimeSlot[ueNodeId];
            // std::cout << "S Time = " << Simulator::Now().GetSeconds() << "ueNodeID = " << ueNodeId << ", OutrageTime = " << ueNodeIdToPerSecDlOutrageTimeSlot[ueNodeId]
            //           << ", Total Time = " << ueNodeIdToPerSecDlTotalTimeSlot[ueNodeId] << ", OR = " << OR << std::endl;
        }
    }
}

void SchedulePerSecUpdateUeFlowStats()
{
    UpdateUeFlowStats();
    Simulator::Schedule(Seconds(1), &SchedulePerSecUpdateUeFlowStats);
}
void UpdateUeOutrageStats(NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); i++)
    {
        uint32_t ueNodeId = ues.Get(i)->GetId();
        ueNodeIdToPerSecDlTotalTimeSlot[ueNodeId] += 1;
        std::cout << "tt" << ues.GetN() << std::endl;
        std::cout << Simulator::Now().GetSeconds() << "ueID " << ueNodeId << ", serving cellid" << ueNodeIdToServingCellIds[ues.Get(i)->GetId()] << ", sinr = " << ueNodeIdToCellIdToSinrs[ueNodeId][ueNodeIdToServingCellIds[ueNodeId]] << std::endl;
        if (ueNodeIdToCellIdToSinrs[ueNodeId][ueNodeIdToServingCellIds[ueNodeId]] < 8)
        {
            ueNodeIdToPerSecDlOutrageTimeSlot[ueNodeId] += 1;
        }
    }
}
void SchedulePerPointOneSecUeUpdateOutrageStats(NodeContainer ues)
{
    UpdateUeOutrageStats(ues);
    Simulator::Schedule(Seconds(0.1), &SchedulePerPointOneSecUeUpdateOutrageStats, ues);
}
QosRequirement
GetUeQosRequirement(uint32_t ueNodeId)
{
    uint32_t choice = ueNodeIdToQosRequirementChoices[ueNodeId];
    return qosRequirementOptions[choice];
}

bool CheckIsNgbr(enum EpsBearer::Qci qci)
{
    bool isNgbr = (std::find(ngbrQciOptions.begin(), ngbrQciOptions.end(), qci) != ngbrQciOptions.end());
    return isNgbr;
}

EpsBearer
GetBearer(QosRequirement qosRequirement)
{
    GbrQosInformation qos;
    qos.gbrDl = qosRequirement.bitrate;
    qos.gbrUl = qos.gbrDl;
    qos.mbrDl = qos.gbrDl;
    qos.mbrUl = qos.gbrUl;
    enum EpsBearer::Qci q = qosRequirement.qci;
    EpsBearer bearer{q, qos};
    return bearer;
}

UeQosRequirementState
GetUeQosRequirementState(uint32_t ueNodeId)
{
    QosRequirement qosRequirement = GetUeQosRequirement(ueNodeId);
    EpsBearer bearer = GetBearer(qosRequirement);
    bool isNgbr = CheckIsNgbr(bearer.qci);
    double requestedBitrate = qosRequirement.bitrate;
    double requestedDelay = 1e-3 * bearer.GetPacketDelayBudgetMs();
    double requestedErrorRate = bearer.GetPacketErrorLossRate();
    UeQosRequirementState state{isNgbr, requestedBitrate, requestedDelay, requestedErrorRate};
    return state;
}

double
NewUeAchievedBitrate(uint32_t ueNodeId)
{
    double achievedBitrate = ueNodeIdToPerSecSumRxBytes[ueNodeId] * 8.0;
    return achievedBitrate;
}

double
NewUeAchievedDelay(uint32_t ueNodeId)
{
    uint32_t perSecNrRxPacket = ueNodeIdToPerSecNrRxPackets[ueNodeId];
    double achievedDelay = (perSecNrRxPacket > 0 ? ueNodeIdToPerSecSumDelays[ueNodeId] / perSecNrRxPacket : 0.0);
    return achievedDelay;
}

double
NewUeAchievedErrorRate(uint32_t ueNodeId)
{
    // double nrOk = ueNodeIdToPerSecNrRxEndOks[ueNodeId];
    // double nrError = ueNodeIdToPerSecNrRxEndErrors[ueNodeId];
    // double nrOkPlusError = nrOk + nrError;
    // double achievedErrorRate = nrOkPlusError > 0 ? nrError / nrOkPlusError : 0;
    // return achievedErrorRate;

    double achievedErrorRate = ueNodeIdToPacketErrorRates[ueNodeId];
    return achievedErrorRate;
}

double
NewBitrateQoeScore(double achievedBitrate, QosRequirement qosRequirement, uint32_t qoeType)
{
    double score = -100.0;
    if (qoeType == 0)
    {
        score = 1.0;
        bool isNgbr = CheckIsNgbr(qosRequirement.qci);
        if (isNgbr)
        {
            return score;
        }
        double requestedBitrate = qosRequirement.bitrate;
        score = 1 / (1 + std::exp(-11 * (achievedBitrate / requestedBitrate - 0.5)));
        return score;
    }
    if (qoeType == 1)
    {
        double requestedBitrate = qosRequirement.bitrate;
        score = achievedBitrate / requestedBitrate;
        return score;
    }
    if (qoeType == 2)
    {
        score = 1.0;
        bool isNgbr = CheckIsNgbr(qosRequirement.qci);
        if (isNgbr)
        {
            return score;
        }
        double requestedBitrate = qosRequirement.bitrate;
        score = 1 / (1 + std::exp(-11 * (achievedBitrate / requestedBitrate - 0.5)));
        return score;
    }
    if (qoeType == 3)
    {
        score = 1.0;
        bool isNgbr = CheckIsNgbr(qosRequirement.qci);
        if (isNgbr)
        {
            return score;
        }
        double requestedBitrate = qosRequirement.bitrate;
        score = 1 / (1 + std::exp(-11 * (achievedBitrate / requestedBitrate - 0.5)));
        return score;
    }
    if (qoeType == 4)
    {
        score = 1.0;
        return score;
    }

    return score;
}

double
NewDelayQoeScore(double achievedDelay, QosRequirement qosRequirement, uint32_t qoeType)
{
    double score = -100.0;
    if (qoeType == 0)
    {
        EpsBearer bearer = GetBearer(qosRequirement);
        // double requestedDelay = 1e-3 * bearer.GetPacketDelayBudgetMs();
        double requestedDelay = 1e-3 * 100;
        score = 1 - 1 / (1 + std::exp(-11 * (achievedDelay / requestedDelay - 1.5)));
        return score;
    }
    if (qoeType == 1)
    {
        score = 1.0;
        return score;
    }
    if (qoeType == 2)
    {
        score = 1.0;
        return score;
    }
    if (qoeType == 3)
    {
        EpsBearer bearer = GetBearer(qosRequirement);
        double requestedDelay = 1e-3 * bearer.GetPacketDelayBudgetMs();
        score = 1 - 1 / (1 + std::exp(-11 * (achievedDelay / requestedDelay - 1.5)));
        return score;
    }
    if (qoeType == 4)
    {
        EpsBearer bearer = GetBearer(qosRequirement);
        double requestedDelay = 1e-3 * bearer.GetPacketDelayBudgetMs();
        score = 1 - 1 / (1 + std::exp(-11 * (achievedDelay / requestedDelay - 1.5)));
        return score;
    }
    return score;
}

double
NewErrorRateQoeScore(double achievedErrorRate, QosRequirement qosRequirement, uint32_t qoeType)
{
    double score = -100.0;
    if (qoeType == 0)
    {
        EpsBearer bearer = GetBearer(qosRequirement);
        double requestedErrorRate = bearer.GetPacketErrorLossRate();
        score = 1 - 1 / (1 + std::exp(-11 * (achievedErrorRate / requestedErrorRate - 1.5)));
        return score;
    }
    if (qoeType == 1)
    {
        score = 1.0;
        return score;
    }
    if (qoeType == 2)
    {
        score = 1.0;
        return score;
    }
    if (qoeType == 3)
    {
        EpsBearer bearer = GetBearer(qosRequirement);
        double requestedErrorRate = bearer.GetPacketErrorLossRate();
        score = 1 - 1 / (1 + std::exp(-11 * (achievedErrorRate / requestedErrorRate - 1.5)));
        return score;
    }
    if (qoeType == 4)
    {
        score = 1.0;
        return score;
    }
    return score;
}

double
NewUeQoeScore(double achievedBitrate, double achievedDelay, double achievedErrorRate, QosRequirement qosRequirement, uint32_t qoeType, uint32_t qtype)
{
    double bitrateQoeScore = NewBitrateQoeScore(achievedBitrate, qosRequirement, qoeType);
    double delayQoeScore = NewDelayQoeScore(achievedDelay, qosRequirement, qoeType);
    double errorRateQoeScore = NewErrorRateQoeScore(achievedErrorRate, qosRequirement, qoeType);
    if (qtype == 4)
    {
        bitrateQoeScore = 1;
        errorRateQoeScore = 1;
        std::cout << 4 << qtype;
        // fgetc(stdin);
    }
    else if (qtype == 2)
    {
        delayQoeScore = 1;
        errorRateQoeScore = 1;
        std::cout << 2 << qtype;
        // fgetc(stdin);
    }
    else
    {
        bitrateQoeScore = 1;
        delayQoeScore = 1;
        std::cout << 1 << qtype;
        // fgetc(stdin);
    }

    double qoeScore = bitrateQoeScore * delayQoeScore * errorRateQoeScore;

    std::cout << "bitrateQoeScore" << bitrateQoeScore << "delayQoeScore" << delayQoeScore << "errorRateQoeScore" << errorRateQoeScore << "qoeScore" << qoeScore << std::endl;
    return qoeScore;
}

UeQoeFlowState
NewUeQoeFlowState(uint32_t ueNodeId, uint32_t qoeType)
{
    QosRequirement qosRequirement = GetUeQosRequirement(ueNodeId);
    // uint32_t qtype = obsUeInfo.at("qtype").at(ueNodeId - 28);
    uint32_t qtype = 4;
    double achievedBitrate = NewUeAchievedBitrate(ueNodeId);
    double achievedDelay = NewUeAchievedDelay(ueNodeId);
    double achievedErrorRate = NewUeAchievedErrorRate(ueNodeId);
    double qoeScore = NewUeQoeScore(achievedBitrate, achievedDelay, achievedErrorRate, qosRequirement, qoeType, qtype);
    UeQoeFlowState state{qoeScore, achievedBitrate, achievedDelay, achievedErrorRate};
    return state;
}

void SetUeNodeIdToUeQoeFlowStates(NodeContainer ues, uint32_t qoeType)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        uint32_t ueNodeId = ues.Get(i)->GetId();
        UeQoeFlowState state = NewUeQoeFlowState(ueNodeId, qoeType);
        ueNodeIdToUeQoeFlowStates[ueNodeId] = state;
    }
}

void SchedulePerSecSetUeNodeIdToUeQoeFlowStates(NodeContainer ues, uint32_t qoeType)
{
    SetUeNodeIdToUeQoeFlowStates(ues, qoeType);
    Simulator::Schedule(Seconds(1), &SchedulePerSecSetUeNodeIdToUeQoeFlowStates, ues, qoeType);
}

void SetEnbNodeIdToNrUsedSubframes(NodeContainer enbs)
{
    for (uint32_t i = 0; i < enbs.GetN(); ++i)
    {
        uint32_t enbNodeId = enbs.Get(i)->GetId();
        auto &dlSchedInfoQue = enbNodeIdToDlSchedInfoQues[enbNodeId];
        uint32_t nrUsedSubframe = 0;
        while (!dlSchedInfoQue.empty())
        {
            dlSchedInfoQue.pop_front();
            nrUsedSubframe++;
        }
        enbNodeIdToNrUsedSubframes[enbNodeId] = nrUsedSubframe;
    }
}

void SchedulePerSecSetEnbNodeIdToNrUsedSubframes(NodeContainer enbs)
{
    SetEnbNodeIdToNrUsedSubframes(enbs);
    Simulator::Schedule(Seconds(1), &SchedulePerSecSetEnbNodeIdToNrUsedSubframes, enbs);
}

void SetEnbNodeIdToPerSecNrDlPhyTxNdis(NodeContainer enbs)
{
    for (uint32_t i = 0; i < enbs.GetN(); ++i)
    {
        uint32_t enbNodeId = enbs.Get(i)->GetId();
        enbNodeIdToPerSecNrDlPhyTxs[enbNodeId] = enbNodeIdToNrDlPhyTxs[enbNodeId] - enbNodeIdToPrevNrDlPhyTxs[enbNodeId];
        enbNodeIdToPrevNrDlPhyTxs[enbNodeId] = enbNodeIdToNrDlPhyTxs[enbNodeId];
        enbNodeIdToPerSecNrDlPhyNdis[enbNodeId] = enbNodeIdToNrDlPhyNdis[enbNodeId] - enbNodeIdToPrevNrDlPhyNdis[enbNodeId];
        enbNodeIdToPrevNrDlPhyNdis[enbNodeId] = enbNodeIdToNrDlPhyNdis[enbNodeId];
    }
}

void SchedulePerSecSetEnbNodeIdToPerSecNrDlPhyTxNdis(NodeContainer enbs)
{
    SetEnbNodeIdToPerSecNrDlPhyTxNdis(enbs);
    Simulator::Schedule(Seconds(1), &SchedulePerSecSetEnbNodeIdToPerSecNrDlPhyTxNdis, enbs);
}

void SetEnbNodeIdToMcsDevDistrs(NodeContainer enbs)
{
    // std::cout << "SetEnbNodeIdToMcsDevDistrs Begin" << std::endl;
    for (uint32_t i = 0; i < enbs.GetN(); ++i)
    {
        uint32_t enbNodeId = enbs.Get(i)->GetId();
        auto &que = enbNodeIdToRntiMcsPairQues[enbNodeId];
        std::map<uint16_t, uint32_t> counts;
        std::map<uint16_t, double> totals;
        std::map<uint16_t, double> squareTotals;
        std::map<uint16_t, double> meanCurrs;
        std::map<uint16_t, double> sCurrs;
        std::map<uint16_t, double> varianceCurrs;
        std::map<uint16_t, double> meanPrevs;
        std::map<uint16_t, double> sPrevs;
        std::map<uint16_t, uint16_t> mins;
        std::map<uint16_t, uint16_t> maxs;
        while (!que.empty())
        {
            // auto &p = que.front ();
            auto p = que.front();
            que.pop_front();
            uint16_t rnti = p.first;
            uint16_t mcs = p.second;
            // std::cout << rnti << "\t" << mcs << "\t" << std::endl;
            ++counts[rnti];
            totals[rnti] += mcs;
            squareTotals[rnti] += mcs * mcs;
            // std::cout << rnti << "\t" << mcs << "\t" << counts[rnti] << "\t" << totals[rnti] << "\t" << squareTotals[rnti] << "\t" << std::endl;
            if (counts[rnti] == 1)
            {
                mins[rnti] = mcs;
                maxs[rnti] = mcs;
                meanCurrs[rnti] = mcs;
                sCurrs[rnti] = 0;
                varianceCurrs[rnti] = sCurrs[rnti];
            }
            else
            {
                meanPrevs[rnti] = meanCurrs[rnti];
                // std::cout << meanPrevs[rnti] << std::endl;
                sPrevs[rnti] = sCurrs[rnti];
                // std::cout << sPrevs[rnti] << std::endl;
                meanCurrs[rnti] = meanPrevs[rnti] + (mcs - meanPrevs[rnti]) / counts[rnti];
                // std::cout << meanCurrs[rnti] << std::endl;
                sCurrs[rnti] = sPrevs[rnti] + (mcs - meanPrevs[rnti]) * (mcs - meanCurrs[rnti]);
                // std::cout << sCurrs[rnti] << std::endl;
                varianceCurrs[rnti] = sCurrs[rnti] / (counts[rnti] - 1);
                // std::cout << varianceCurrs[rnti] << std::endl;
            }
            // std::cout << "is empty?" << "\t" << que.empty() << std::endl;
        }
        // std::cout << "SetEnbNodeIdToMcsDevDistrs mid" << i << std::endl;
        std::vector<uint32_t> mcsDevDistr = {0, 0, 0};
        for (const auto &p : varianceCurrs)
        {
            // std::cout << p.first << ": ";
            // std::cout << std::sqrt(p.second) << "\t";
            auto dev = std::sqrt(p.second);
            if (dev < 1.0)
            {
                ++mcsDevDistr[0];
            }
            else if (1.0 <= dev && dev < 3)
            {
                ++mcsDevDistr[1];
            }
            else
            {
                ++mcsDevDistr[2];
            }
        }
        enbNodeIdToMcsDevDistrs[enbNodeId] = mcsDevDistr;
    }
    std::cout << "SetEnbNodeIdToMcsDevDistrs End" << std::endl;
}

void SchedulePerSecSetEnbNodeIdToMcsDevDistrs(NodeContainer enbs)
{
    SetEnbNodeIdToMcsDevDistrs(enbs);
    Simulator::Schedule(Seconds(1), &SchedulePerSecSetEnbNodeIdToMcsDevDistrs, enbs);
}

// void
// SetUeImsiToMeanMcss (NodeContainer ues)
//{
//   for (uint32_t i = 0; i < ues.GetN (); ++i)
//     {
//       uint32_t imsi = ues.Get (i)->GetDevice (1)->GetObject<LteUeNetDevice> ()->GetImsi ();
//       auto &que = ueImsiToEnbNodeIdMcsPairQues[imsi];
//       std::map<uint32_t, uint32_t> counts;
//       std::map<uint32_t, double> meanCurrs;
//       std::map<uint32_t, double> meanPrevs;
//       uint32_t enbNodeId = 0;
//       while (!que.empty ())
//         {
//           auto p = que.front ();
//           que.pop_front ();
//           enbNodeId = p.first;
//           uint32_t mcs = p.second;
//           ++counts[enbNodeId];
//           if (counts[enbNodeId] == 1)
//             {
//               meanCurrs[enbNodeId] = mcs;
//             }
//           else
//             {
//               meanPrevs[enbNodeId] = meanCurrs[enbNodeId];
//               meanCurrs[enbNodeId] = meanPrevs[enbNodeId] + (mcs - meanPrevs[enbNodeId]) / counts[enbNodeId];
//             }
//         }
//       ueImsiToMeanMcss[imsi] = meanCurrs[enbNodeId];
//     }
// }

// void
// SchedulePerSecSetUeImsiToMeanMcss (NodeContainer ues)
//{
//   SetUeImsiToMeanMcss (ues);
//   Simulator::Schedule (Seconds (1), &SchedulePerSecSetUeImsiToMeanMcss, ues);
// }

void SetCellIdToRntiToMeanMcss()
{
    for (uint32_t i = 0; i < cellIds.size(); ++i)
    {
        uint32_t cellId = cellIds[i];
        auto &que = cellIdToRntiMcsPairQues[cellId];
        std::map<uint32_t, uint32_t> counts;
        std::map<uint32_t, double> meanCurrs;
        std::map<uint32_t, double> meanPrevs;
        while (!que.empty())
        {
            auto p = que.front();
            que.pop_front();
            uint32_t rnti = p.first;
            uint32_t mcs = p.second;
            ++counts[rnti];
            if (counts[rnti] == 1)
            {
                meanCurrs[rnti] = mcs;
            }
            else
            {
                meanPrevs[rnti] = meanCurrs[rnti];
                meanCurrs[rnti] = meanPrevs[rnti] + (mcs - meanPrevs[rnti]) / counts[rnti];
            }
        }
        cellIdToRntiToMeanMcss[cellId] = meanCurrs;
    }
}

void SchedulePerSecSetCellIdToRntiToMeanMcss()
{
    SetCellIdToRntiToMeanMcss();
    Simulator::Schedule(Seconds(1), &SchedulePerSecSetCellIdToRntiToMeanMcss);
}

void SetUeNodeIdToPacketErrorRates(NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        uint32_t ueNodeId = ue->GetId();
        uint64_t imsi = ue->GetDevice(1)->GetObject<LteUeNetDevice>()->GetImsi();
        double dlTxPackets = ueImsiToDlTxPackets[imsi] - ueImsiToPrevDlTxPackets[imsi];
        ueImsiToPrevDlTxPackets[imsi] = ueImsiToDlTxPackets[imsi];
        double dlRxPackets = ueImsiToDlRxPackets[imsi] - ueImsiToPrevDlRxPackets[imsi];
        ueImsiToPrevDlRxPackets[imsi] = ueImsiToDlRxPackets[imsi];
        double packetErrorRate = (dlTxPackets - dlRxPackets) / dlTxPackets;
        if (packetErrorRate < 0)
        {
            packetErrorRate = 0.0;
        }
        ueNodeIdToPacketErrorRates[ueNodeId] = packetErrorRate;
        // std::cout << ueNodeId << "\t" << packetErrorRate << "\t" << std::endl;
    }
}

void SchedulePerSecSetUeNodeIdToPacketErrorRates(NodeContainer ues)
{
    SetUeNodeIdToPacketErrorRates(ues);
    Simulator::Schedule(Seconds(1), &SchedulePerSecSetUeNodeIdToPacketErrorRates, ues);
}

void PrintEnbInfo(std::ofstream &ofs,
                  NodeContainer enbs,
                  const std::map<std::string, std::vector<double>> &enbInfo)
{
    const std::vector<double> &xs = enbInfo.at("x");
    const std::vector<double> &ys = enbInfo.at("y");
    const std::vector<double> &zs = enbInfo.at("z");
    const std::vector<double> &bandwidthNrRbs = enbInfo.at("bandwidthNrRb");
    const std::vector<double> &subBandOffsets = enbInfo.at("subBandOffset");
    const std::vector<double> &subBandwidthNrRbs = enbInfo.at("subBandwidthNrRb");
    const std::vector<double> &txPowerDbms = enbInfo.at("txPowerDbm");
    const std::vector<double> &schedulerChoices = enbInfo.at("schedulerChoice");
    for (uint32_t i = 0; i < enbs.GetN(); ++i)
    {
        ofs << cellIds[i] << "\t"
            << xs[i] << "\t"
            << ys[i] << "\t"
            << zs[i] << "\t"
            << bandwidthNrRbs[i] << "\t"
            << subBandOffsets[i] << "\t"
            << subBandwidthNrRbs[i] << "\t"
            << txPowerDbms[i] << "\t"
            << schedulerChoices[i] << "\t"
            << std::endl;
    }
}

void PrintUeInfo(std::ofstream &ofs,
                 NodeContainer ues,
                 const std::map<std::string, std::vector<double>> &ueInfo)
{
    const std::vector<double> &ueQosChoices = ueInfo.at("qosRequirementChoice");
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        uint32_t ueNodeId = ues.Get(i)->GetId();
        uint32_t choice = (uint32_t)ueQosChoices[i];
        GbrQosInformation qos;
        qos.gbrDl = qosRequirementOptions[choice].bitrate;
        qos.gbrUl = qos.gbrDl;
        qos.mbrDl = qos.gbrDl;
        qos.mbrUl = qos.gbrUl;
        enum EpsBearer::Qci q = qosRequirementOptions[choice].qci;
        EpsBearer bearer{q, qos};

        ofs << ueNodeId << "\t"
            << choice << "\t"
            << qosRequirementOptions[choice].bitrate << "\t"
            << qosRequirementOptions[choice].qci << "\t"
            << bearer.GetPacketDelayBudgetMs() << "\t"
            << bearer.GetPacketErrorLossRate() << "\t"
            << std::endl;
    }
}

void PrintUeImsi(std::ofstream &ofs, NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        uint32_t ueNodeId = ue->GetId();
        uint64_t imsi = ue->GetDevice(1)->GetObject<LteUeNetDevice>()->GetImsi();
        ofs << ueNodeId << "\t"
            << imsi << "\t"
            << std::endl;
    }
}

void PrintNodePos(std::ofstream &ofs, Ptr<Node> node)
{
    Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
    Vector pos = mob->GetPosition();
    ofs << Simulator::Now().GetSeconds() << "\t"
        << node->GetId() << "\t"
        << pos.x << "\t"
        << pos.y << "\t"
        << pos.z << "\t"
        << std::endl;
}

void SchedulePerSecPrintNodePoss(std::ofstream *ofs, NodeContainer nodes)
{
    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        Ptr<Node> node = nodes.Get(i);
        PrintNodePos(*ofs, node);
    }
    Simulator::Schedule(Seconds(1), &SchedulePerSecPrintNodePoss, ofs, nodes);
}

std::map<uint16_t, uint32_t>
ComputeCellIdToNrServingUes()
{
    std::map<uint16_t, uint32_t> cellIdToNrServingUes;
    for (const auto &m : ueNodeIdToServingCellIds)
    {
        cellIdToNrServingUes[m.second]++;
    }
    return cellIdToNrServingUes;
}

void PrintCellNrServingUe(std::ofstream &ofs,
                          std::map<uint16_t, uint32_t> &cellIdToNrServingUes,
                          uint32_t cellIdIndex)
{
    ofs << Simulator::Now().GetSeconds() << "\t"
        << cellIds[cellIdIndex] << "\t"
        << cellIdToNrServingUes[cellIds[cellIdIndex]] << "\t"
        << std::endl;
}

void SchedulePerSecPrintEachCellNrServingUe(std::ofstream *ofs)
{
    std::map<uint16_t, uint32_t> cellIdToNrServingUes = ComputeCellIdToNrServingUes();
    for (uint32_t i = 0; i < cellIds.size(); ++i)
    {
        PrintCellNrServingUe(*ofs, cellIdToNrServingUes, i);
    }
    Simulator::Schedule(Seconds(1), &SchedulePerSecPrintEachCellNrServingUe, ofs);
}

void PrintUeRxByte(std::ofstream &ofs, Ptr<Node> ue)
{
    uint32_t ueNodeId = ue->GetId();
    ofs << Simulator::Now().GetSeconds() << "\t"
        << ueNodeId << "\t"
        << ueNodeIdToTotalRxBytes[ueNodeId] << "\t"
        << std::endl;
}

void SchedulePerSecPrintUeRxBytes(std::ofstream *ofs, NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        PrintUeRxByte(*ofs, ue);
    }
    Simulator::Schedule(Seconds(1), &SchedulePerSecPrintUeRxBytes, ofs, ues);
}

void PrintUeNrRlf(std::ofstream &ofs, Ptr<Node> ue)
{
    uint32_t ueNodeId = ue->GetId();
    ofs << Simulator::Now().GetSeconds() << "\t"
        << ueNodeId << "\t"
        << ueNodeIdToTotalNrRlfs[ueNodeId] << "\t"
        << std::endl;
}

void SchedulePerSecPrintUeNrRlfs(std::ofstream *ofs, NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        PrintUeNrRlf(*ofs, ue);
    }
    Simulator::Schedule(Seconds(1), &SchedulePerSecPrintUeNrRlfs, ofs, ues);
}

void PrintUeNrHandover(std::ofstream &ofs, Ptr<Node> ue)
{
    uint32_t ueNodeId = ue->GetId();
    ofs << Simulator::Now().GetSeconds() << "\t"
        << ueNodeId << "\t"
        << ueNodeIdToTotalNrHandovers[ueNodeId] << "\t"
        << std::endl;
}

void SchedulePerSecPrintUeNrHandovers(std::ofstream *ofs, NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        PrintUeNrHandover(*ofs, ue);
    }
    Simulator::Schedule(Seconds(1), &SchedulePerSecPrintUeNrHandovers, ofs, ues);
}

void PrintUeNrPingpong(std::ofstream &ofs, Ptr<Node> ue)
{
    uint32_t ueNodeId = ue->GetId();
    ofs << Simulator::Now().GetSeconds() << "\t"
        << ueNodeId << "\t"
        << ueNodeIdToTotalNrPingpongs[ueNodeId] << "\t"
        << std::endl;
}

void SchedulePerSecPrintUeNrPingpongs(std::ofstream *ofs, NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        PrintUeNrPingpong(*ofs, ue);
    }
    Simulator::Schedule(Seconds(1), &SchedulePerSecPrintUeNrPingpongs, ofs, ues);
}

void PrintUeServingCellId(std::ofstream &ofs, Ptr<Node> ue)
{
    uint32_t ueNodeId = ue->GetId();
    ofs << Simulator::Now().GetSeconds() << "\t"
        << ueNodeId << "\t"
        << ueNodeIdToServingCellIds[ueNodeId] << "\t"
        << std::endl;
}

void SchedulePerSecPrintUeServingCellIds(std::ofstream *ofs, NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        PrintUeServingCellId(*ofs, ue);
    }
    Simulator::Schedule(Seconds(1), &SchedulePerSecPrintUeServingCellIds, ofs, ues);
}

void SchedulePer25msPrintUeServingCellIds(std::ofstream *ofs, NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        PrintUeServingCellId(*ofs, ue);
    }
    Simulator::Schedule(Seconds(0.025), &SchedulePer25msPrintUeServingCellIds, ofs, ues);
}

void PrintUeQoeFlowState(std::ofstream &ofs, Ptr<Node> ue)
{
    uint32_t ueNodeId = ue->GetId();
    auto &state = ueNodeIdToUeQoeFlowStates[ueNodeId];
    auto qosState = GetUeQosRequirementState(ueNodeId);
    ofs << Simulator::Now().GetSeconds() << "\t"
        << ueNodeId << "\t"
        << state.qoeScore << "\t"
        << state.achievedBitrate << "\t"
        << state.achievedDelay << "\t"
        << state.achievedErrorRate << "\t"
        << qosState.isNgbr << "\t"
        << qosState.requestedBitrate << "\t"
        << qosState.requestedDelay << "\t"
        << qosState.requestedErrorRate << "\t"
        << std::endl;
}

void SchedulePerSecPrintUeQoeFlowStates(std::ofstream *ofs, NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        PrintUeQoeFlowState(*ofs, ue);
    }
    Simulator::Schedule(Seconds(1), &SchedulePerSecPrintUeQoeFlowStates, ofs, ues);
}

void PrintFlowStats(std::ofstream &ofs)
{
    auto stats = monitor->GetFlowStats();
    for (auto i = stats.cbegin(); i != stats.cend(); ++i)
    {
        uint32_t ip = classifier->FindFlow(i->first).destinationAddress.Get();
        uint32_t ueNodeId = ipv4AddressToUeNodeIds[ip];
        ofs << Simulator::Now().GetSeconds() << "\t"
            << ueNodeId << "\t"
            << i->second.delaySum.GetSeconds() << "\t"
            << i->second.rxPackets << "\t"
            << i->second.rxBytes << "\t"
            << ueNodeIdToNrPhyRxEndOks[ueNodeId] << "\t"
            << ueNodeIdToNrPhyRxEndErrors[ueNodeId] << "\t"
            << std::endl;
    }
}

void SchedulePer25msPrintFlowStats(std::ofstream *ofs)
{
    PrintFlowStats(*ofs);
    Simulator::Schedule(Seconds(0.025), &SchedulePer25msPrintFlowStats, ofs);
}

void PrintUeServingCellIdRsrqRsrpSinr(std::ofstream &ofs, Ptr<Node> ue)
{
    uint32_t ueNodeId = ue->GetId();
    uint16_t cellId = ueNodeIdToServingCellIds[ueNodeId];
    double rsrq = ueNodeIdToCellIdToRsrqs[ueNodeId][cellId];
    double rsrp = ueNodeIdToCellIdToRsrps[ueNodeId][cellId];
    double sinr = ueNodeIdToCellIdToSinrs[ueNodeId][cellId];
    ofs << Simulator::Now().GetSeconds() << "\t"
        << ueNodeId << "\t"
        << cellId << "\t"
        << rsrq << "\t"
        << rsrp << "\t"
        << sinr << "\t"
        << std::endl;
}

void SchedulePer25msPrintUeServingCellIdRsrqRsrpSinrs(std::ofstream *ofs, NodeContainer ues)
{
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ptr<Node> ue = ues.Get(i);
        PrintUeServingCellIdRsrqRsrpSinr(*ofs, ue);
    }
    Simulator::Schedule(Seconds(0.025), &SchedulePer25msPrintUeServingCellIdRsrqRsrpSinrs, ofs, ues);
}

std::map<uint16_t, std::map<uint64_t, uint32_t>>
NewCellIdToBitrateDistrs(NodeContainer *obsUes)
{
    std::map<uint16_t, std::map<uint64_t, uint32_t>> cellIdToBitrateDistrs;
    for (uint32_t i = 0; i < obsUes->GetN(); ++i)
    {
        uint32_t ueNodeId = obsUes->Get(i)->GetId();
        uint16_t cellId = ueNodeIdToServingCellIds[ueNodeId];
        UeQosRequirementState state = GetUeQosRequirementState(ueNodeId);
        uint64_t bitrate = state.requestedBitrate;
        cellIdToBitrateDistrs[cellId][bitrate]++;
    }
    return cellIdToBitrateDistrs;
}

void SetUeTriggerPrevFlowStats(uint32_t ueNodeId)
{
    auto stats = monitor->GetFlowStats();
    for (auto i = stats.cbegin(); i != stats.cend(); ++i)
    {
        uint32_t ip = classifier->FindFlow(i->first).destinationAddress.Get();
        if (ueNodeId == ipv4AddressToUeNodeIds[ip])
        {
            ueNodeIdToTriggerPrevSumDelays[ueNodeId] = i->second.delaySum.GetSeconds();
            ueNodeIdToTriggerPrevNrRxPackets[ueNodeId] = i->second.rxPackets;
            ueNodeIdToTriggerPrevSumRxBytes[ueNodeId] = i->second.rxBytes;
            ueNodeIdToTriggerPrevNrTxPackets[ueNodeId] = i->second.txPackets;
            ueNodeIdToTriggerPrevNrRxEndOks[ueNodeId] = ueNodeIdToNrRxEndOks[ueNodeId];
            ueNodeIdToTriggerPrevNrRxEndErrors[ueNodeId] = ueNodeIdToNrRxEndErrors[ueNodeId];
            nrPrevPacketSend[ueNodeId - 5] = nrPacketSend[ueNodeId - 5];
            nrPrevPacketReceive[ueNodeId - 5] = nrPacketReceive[ueNodeId - 5];
            break;
        }
    }
}

void SetUeTriggerOneSecFlowStats(uint32_t ueNodeId)
{
    auto stats = monitor->GetFlowStats();
    for (auto i = stats.cbegin(); i != stats.cend(); ++i)
    {
        uint32_t ip = classifier->FindFlow(i->first).destinationAddress.Get();
        if (ueNodeId == ipv4AddressToUeNodeIds[ip])
        {
            ueNodeIdToTriggerOneSecSumDelays[ueNodeId] = i->second.delaySum.GetSeconds() - ueNodeIdToTriggerPrevSumDelays[ueNodeId];
            ueNodeIdToTriggerOneSecNrRxPackets[ueNodeId] = i->second.rxPackets - ueNodeIdToTriggerPrevNrRxPackets[ueNodeId];
            ueNodeIdToTriggerOneSecSumRxBytes[ueNodeId] = i->second.rxBytes - ueNodeIdToTriggerPrevSumRxBytes[ueNodeId];
            ueNodeIdToTriggerOneSecNrTxPackets[ueNodeId] = nrPacketSend[ueNodeId - 5] - nrPrevPacketSend[ueNodeId - 5];
            ueNodeIdToTriggerOneSecNrRxEndOks[ueNodeId] = nrPacketReceive[ueNodeId - 5] - nrPrevPacketReceive[ueNodeId - 5];
            ueNodeIdToTriggerOneSecNrRxEndErrors[ueNodeId] = ueNodeIdToTriggerOneSecNrTxPackets[ueNodeId] - ueNodeIdToTriggerOneSecNrRxEndOks[ueNodeId];
            if (ueNodeIdToTriggerOneSecNrRxEndErrors[ueNodeId] < 0)
            {
                ueNodeIdToTriggerOneSecNrRxEndErrors[ueNodeId] += 1;
            }
            break;
        }
    }
}

void ScheduleObsUePeriodicTrigger(Ptr<OpenGymInterface> openGymIface, NodeContainer *obsUes, uint32_t obsUeIndex, uint32_t triggerIntervalMilliSec)
{
    currentHandoverUeIndex = obsUeIndex;
    uint32_t ueNodeId = obsUes->Get(obsUeIndex)->GetId();
    uint16_t servingCellId = ueNodeIdToServingCellIds[ueNodeId];
    double interval = 0.001 * obsUes->GetN() * triggerIntervalMilliSec;

    Simulator::Schedule(Seconds(0.001), &SetUeTriggerPrevFlowStats, ueNodeId);
    Simulator::Schedule(Seconds(0.999), &SetUeTriggerOneSecFlowStats, ueNodeId);

    if (servingCellId != 0)
    {
        // openGymIface->NotifyCurrentState();
        std::cout << Simulator::Now().GetSeconds() << std::endl;
        // fgetc(stdin);
        // if (Simulator::Now().GetSeconds() >= 1.9 && Simulator::Now().GetSeconds() < 2)
        // {
        //     std::cout << Simulator::Now().GetSeconds() << std::endl;
        //     fgetc(stdin);
        //     openGymIface->NotifyCurrentState();
        // }
    }

    Simulator::Schedule(Seconds(interval), &ScheduleObsUePeriodicTrigger, openGymIface, obsUes, obsUeIndex, triggerIntervalMilliSec);
}

Ptr<OpenGymSpace>
GetG6ActSpace(uint32_t nrEnb)
{
    Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace>(nrEnb);
    return space;
}

Ptr<OpenGymSpace>
GetG6ObsSpace(uint32_t nrEnb, NodeContainer *obsUes)
{
    std::string dtype = TypeNameGet<float>();
    std::vector<uint32_t> rsrqShape = {
        nrEnb,
    };
    Ptr<OpenGymBoxSpace> rsrqSpace = CreateObject<OpenGymBoxSpace>(-100, 100, rsrqShape, dtype);
    std::vector<uint32_t> rssirShape = {
        nrEnb,
    };
    Ptr<OpenGymBoxSpace> rssirSpace = CreateObject<OpenGymBoxSpace>(-100, 100, rssirShape, dtype);
    std::vector<uint32_t> nrUeShape = {
        nrEnb,
    };
    Ptr<OpenGymBoxSpace> nrUeSpace = CreateObject<OpenGymBoxSpace>(0, 1000, nrUeShape, dtype);
    std::vector<uint32_t> cellIdShape = {
        nrEnb,
    };
    Ptr<OpenGymBoxSpace> cellIdSpace = CreateObject<OpenGymBoxSpace>(0, 1000, cellIdShape, dtype);
    uint32_t nrObsUe = obsUes->GetN();
    std::vector<uint32_t> servCellIdShape = {
        nrObsUe,
    };
    Ptr<OpenGymBoxSpace> servCellIdSpace = CreateObject<OpenGymBoxSpace>(0, 1000, servCellIdShape, dtype);
    std::vector<uint32_t> servCellRsrqShape = {
        nrObsUe,
    };
    Ptr<OpenGymBoxSpace> servCellRsrqSpace = CreateObject<OpenGymBoxSpace>(-100, 100, servCellRsrqShape, dtype);
    std::vector<uint32_t> servCellSirShape = {
        nrObsUe,
    };
    Ptr<OpenGymBoxSpace> servCellSirSpace = CreateObject<OpenGymBoxSpace>(-100, 100, servCellSirShape, dtype);
    Ptr<OpenGymDiscreteSpace> hoUeIndexSpace = CreateObject<OpenGymDiscreteSpace>(1000);
    Ptr<OpenGymDiscreteSpace> servCellIndexSpace = CreateObject<OpenGymDiscreteSpace>(nrEnb);
    std::vector<uint32_t> simTimeShape = {
        1,
    };
    Ptr<OpenGymBoxSpace> simTimeSpace = CreateObject<OpenGymBoxSpace>(0, 100000, simTimeShape, dtype);
    Ptr<OpenGymDictSpace> space = CreateObject<OpenGymDictSpace>();
    space->Add("rsrq", rsrqSpace);
    space->Add("rssir", rssirSpace);
    space->Add("nrUe", nrUeSpace);
    space->Add("cellId", cellIdSpace);
    space->Add("servCellId", servCellIdSpace);
    space->Add("servCellRsrq", servCellRsrqSpace);
    space->Add("servCellSir", servCellSirSpace);
    space->Add("hoUeIndex", hoUeIndexSpace);
    space->Add("servCellIndex", servCellIndexSpace);
    space->Add("simTime", simTimeSpace);
    return space;
}

Ptr<OpenGymDataContainer>
GetG6Obs(uint32_t nrEnb, NodeContainer *obsUes)
{
    uint32_t ueNodeId = obsUes->Get(currentHandoverUeIndex)->GetId();
    uint16_t servingCellId = ueNodeIdToServingCellIds[ueNodeId];

    std::vector<uint32_t> rsrqShape = {
        nrEnb,
    };
    Ptr<OpenGymBoxContainer<float>> rsrqBox = CreateObject<OpenGymBoxContainer<float>>(rsrqShape);
    for (uint32_t i = 0; i < nrEnb; ++i)
    {
        float rsrq = ueNodeIdToCellIdToRsrqs[ueNodeId][cellIds[i]];
        rsrqBox->AddValue(rsrq < -100 ? -100 : rsrq);
    }

    std::vector<uint32_t> rssirShape = {
        nrEnb,
    };
    Ptr<OpenGymBoxContainer<float>> rssirBox = CreateObject<OpenGymBoxContainer<float>>(rssirShape);
    std::map<uint32_t, double> subBandOffsetToInterferences;
    for (uint32_t i = 0; i < nrEnb; ++i)
    {
        uint32_t subBandOffset = cellSubBandOffsets[i];
        double rsrpDbm = ueNodeIdToCellIdToRsrps[ueNodeId][cellIds[i]];
        double rsrpMw = std::pow(10.0, rsrpDbm / 10.0);
        subBandOffsetToInterferences[subBandOffset] += rsrpMw;
    }
    for (uint32_t i = 0; i < nrEnb; ++i)
    {
        uint32_t subBandOffset = cellSubBandOffsets[i];
        double rsrpDbm = ueNodeIdToCellIdToRsrps[ueNodeId][cellIds[i]];
        double rsrpMw = std::pow(10.0, rsrpDbm / 10.0);
        double sir = (rsrpMw) / (subBandOffsetToInterferences[subBandOffset] - rsrpMw);
        double sirDb = 10.0 * std::log10(sir);
        rssirBox->AddValue(sirDb > 100 ? 100 : sirDb < -100 ? -100
                                                            : sirDb);
    }

    std::vector<uint32_t> nrUeShape = {
        nrEnb,
    };
    Ptr<OpenGymBoxContainer<float>> nrUeBox = CreateObject<OpenGymBoxContainer<float>>(nrUeShape);
    auto cellIdToNrUes = ComputeCellIdToNrServingUes();
    for (uint32_t i = 0; i < nrEnb; ++i)
    {
        float nrUe = cellIdToNrUes[cellIds[i]];
        nrUeBox->AddValue(nrUe);
    }

    std::vector<uint32_t> cellIdShape = {
        nrEnb,
    };
    Ptr<OpenGymBoxContainer<float>> cellIdBox = CreateObject<OpenGymBoxContainer<float>>(cellIdShape);
    for (uint32_t i = 0; i < nrEnb; ++i)
    {
        cellIdBox->AddValue(cellIds[i]);
    }

    uint32_t nrObsUe = obsUes->GetN();
    std::vector<uint32_t> servCellIdShape = {
        nrObsUe,
    };
    Ptr<OpenGymBoxContainer<float>> servCellIdBox = CreateObject<OpenGymBoxContainer<float>>(servCellIdShape);
    for (uint32_t i = 0; i < nrObsUe; ++i)
    {
        uint32_t ueNodeId = obsUes->Get(i)->GetId();
        uint16_t cellId = ueNodeIdToServingCellIds[ueNodeId];
        servCellIdBox->AddValue(cellId);
    }

    std::vector<uint32_t> servCellRsrqShape = {
        nrObsUe,
    };
    Ptr<OpenGymBoxContainer<float>> servCellRsrqBox = CreateObject<OpenGymBoxContainer<float>>(servCellRsrqShape);
    for (uint32_t i = 0; i < nrObsUe; ++i)
    {
        uint32_t ueNodeId = obsUes->Get(i)->GetId();
        uint16_t cellId = ueNodeIdToServingCellIds[ueNodeId];
        double rsrq = ueNodeIdToCellIdToRsrqs[ueNodeId][cellId];
        servCellRsrqBox->AddValue(rsrq);
    }

    std::vector<uint32_t> servCellSirShape = {
        nrObsUe,
    };
    Ptr<OpenGymBoxContainer<float>> servCellSirBox = CreateObject<OpenGymBoxContainer<float>>(servCellSirShape);
    for (uint32_t i = 0; i < nrObsUe; ++i)
    {
        uint32_t ueNodeId = obsUes->Get(i)->GetId();
        uint16_t servCellId = ueNodeIdToServingCellIds[ueNodeId];
        if (servCellId == 0)
        {
            servCellSirBox->AddValue(-100);
            continue;
        }
        uint32_t servCellIndex = 0;
        while (servCellId != cellIds[servCellIndex])
        {
            servCellIndex++;
        }
        uint32_t servCellSubBandOffset = cellSubBandOffsets[servCellIndex];
        double interferenceMw = 0.0;
        double signalMw = 0.0;
        for (uint32_t j = 0; j < nrEnb; ++j)
        {
            if (servCellSubBandOffset != cellSubBandOffsets[j])
            {
                continue;
            }
            double rsrpDbm = ueNodeIdToCellIdToRsrps[ueNodeId][cellIds[j]];
            double rsrpMw = std::pow(10.0, rsrpDbm / 10.0);
            if (servCellId == cellIds[j])
            {
                signalMw = rsrpMw;
            }
            else
            {
                interferenceMw += rsrpMw;
            }
        }
        double sir = 10.0 * std::log10(signalMw / interferenceMw);
        servCellSirBox->AddValue(sir);
    }

    Ptr<OpenGymDiscreteContainer> hoUeIndexBox = CreateObject<OpenGymDiscreteContainer>();
    hoUeIndexBox->SetValue(currentHandoverUeIndex);

    Ptr<OpenGymDiscreteContainer> servCellIndexBox = CreateObject<OpenGymDiscreteContainer>();
    for (uint32_t i = 0; i < nrEnb; ++i)
    {
        if (cellIds[i] == servingCellId)
        {
            servCellIndexBox->SetValue(i);
        }
    }

    std::vector<uint32_t> simTimeShape = {
        1,
    };
    Ptr<OpenGymBoxContainer<float>> simTimeBox = CreateObject<OpenGymBoxContainer<float>>(simTimeShape);
    simTimeBox->AddValue(Simulator::Now().GetSeconds());
    Ptr<OpenGymDictContainer> box = CreateObject<OpenGymDictContainer>();
    box->Add("rsrq", rsrqBox);
    box->Add("rssir", rssirBox);
    box->Add("nrUe", nrUeBox);
    box->Add("cellId", cellIdBox);
    box->Add("servCellId", servCellIdBox);
    box->Add("servCellRsrq", servCellRsrqBox);
    box->Add("servCellSir", servCellSirBox);
    box->Add("hoUeIndex", hoUeIndexBox);
    box->Add("servCellIndex", servCellIndexBox);
    box->Add("simTime", simTimeBox);
    return box;
}

double
NewUeTriggerOneSecAchievedBitrate(uint32_t ueNodeId)
{
    double bitrate = ueNodeIdToTriggerOneSecSumRxBytes[ueNodeId] * 8.0;
    return bitrate;
}

double
NewUeTriggerOneSecAchievedDelay(uint32_t ueNodeId)
{
    double delay = ueNodeIdToDelayPerSecond[ueNodeId] * 10e-9;
    return delay;
}

double
NewUeTriggerOneSecAchievedErrorRate(uint32_t ueNodeId)
{
    double nrOk = ueNodeIdToTriggerOneSecNrRxEndOks[ueNodeId];
    double nrError = ueNodeIdToTriggerOneSecNrRxEndErrors[ueNodeId];
    double nrOkPlusError = nrOk + nrError;
    double errorRate = (nrOkPlusError > 0) ? (nrError / nrOkPlusError) : 0.0;
    return errorRate;
}

// opengym
Ptr<OpenGymSpace>
GetG12ActSpace(uint32_t nrEnb)
{
    Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace>(nrEnb);
    return space;
}

Ptr<OpenGymSpace>
GetG12ObsSpace(uint32_t nrEnb, NodeContainer *obsUes)
{
    std::string dtype = TypeNameGet<float>();
    uint32_t nrObsUe = obsUes->GetN();
    // Ptr<OpenGymBoxSpace> outRageRatio = Create<OpenGymBoxSpace>(0, 1, std::vector<uint32_t>({
    //                                                                       nrEnb,
    //                                                                   }),
    //                                                             dtype);
    // PtrPtr<OpenGymBoxSpace> AverageDelay = Create<OpenGymBoxSpace>(0, 1, std::vector<uint32_t>({
    //                                                                          nrEnb,
    //                                                                      }),
    //                                                                dtype);
    Ptr<OpenGymBoxSpace> bitrateDemandSpace = CreateObject<OpenGymBoxSpace>(0, 1000000000, std::vector<uint32_t>({
                                                                                               nrObsUe,
                                                                                           }),
                                                                            dtype);
    Ptr<OpenGymBoxSpace> servCellIdSpace = CreateObject<OpenGymBoxSpace>(0, 1000, std::vector<uint32_t>({
                                                                                      nrObsUe,
                                                                                  }),
                                                                         dtype);
    Ptr<OpenGymBoxSpace> mcsSpace = CreateObject<OpenGymBoxSpace>(0, 100, std::vector<uint32_t>({
                                                                              nrObsUe,
                                                                          }),
                                                                  dtype);
    Ptr<OpenGymBoxSpace> rsrqSpace = CreateObject<OpenGymBoxSpace>(-100, 100, std::vector<uint32_t>({
                                                                                  nrObsUe,
                                                                                  nrEnb,
                                                                              }),
                                                                   dtype);
    Ptr<OpenGymBoxSpace> rsrpSpace = CreateObject<OpenGymBoxSpace>(-100, 100, std::vector<uint32_t>({
                                                                                  nrObsUe,
                                                                                  nrEnb,
                                                                              }),
                                                                   dtype);
    Ptr<OpenGymBoxSpace> newDataRatioSpace = CreateObject<OpenGymBoxSpace>(0, 1, std::vector<uint32_t>({
                                                                                     nrEnb,
                                                                                 }),
                                                                           dtype);
    Ptr<OpenGymBoxSpace> subBandOffsetSpace = CreateObject<OpenGymBoxSpace>(0, 1000, std::vector<uint32_t>({
                                                                                         nrEnb,
                                                                                     }),
                                                                            dtype);
    Ptr<OpenGymBoxSpace> cellIdSpace = CreateObject<OpenGymBoxSpace>(0, 1000, std::vector<uint32_t>({
                                                                                  nrEnb,
                                                                              }),
                                                                     dtype);
    Ptr<OpenGymDiscreteSpace> hoUeIndexSpace = CreateObject<OpenGymDiscreteSpace>(1000);
    Ptr<OpenGymDiscreteSpace> servCellIndexSpace = CreateObject<OpenGymDiscreteSpace>(nrEnb);
    Ptr<OpenGymBoxSpace> simTimeSpace = CreateObject<OpenGymBoxSpace>(0, 100000, std::vector<uint32_t>({
                                                                                     1,
                                                                                 }),
                                                                      dtype);
    Ptr<OpenGymBoxSpace> qoeSpace = CreateObject<OpenGymBoxSpace>(0, 1, std::vector<uint32_t>({
                                                                            nrObsUe,
                                                                        }),
                                                                  dtype);
    Ptr<OpenGymBoxSpace> enbAverageDelaySpace = CreateObject<OpenGymBoxSpace>(0, 1, std::vector<uint32_t>({
                                                                                        nrEnb,
                                                                                    }),
                                                                              dtype);
    Ptr<OpenGymDictSpace> space = CreateObject<OpenGymDictSpace>();
    space->Add("bitrateDemand", bitrateDemandSpace);
    space->Add("servCellId", servCellIdSpace);
    space->Add("mcs", mcsSpace);
    space->Add("rsrq", rsrqSpace);
    space->Add("rsrp", rsrpSpace);
    space->Add("newDataRatio", newDataRatioSpace);
    space->Add("subBandOffset", subBandOffsetSpace);
    space->Add("cellId", cellIdSpace);
    space->Add("hoUeIndex", hoUeIndexSpace);
    space->Add("servCellIndex", servCellIndexSpace);
    space->Add("simTime", simTimeSpace);
    space->Add("qoe", qoeSpace);
    space->Add("enbAverageDelay", enbAverageDelaySpace);
    return space;
}

Ptr<OpenGymDataContainer>
GetG12Obs(uint32_t nrEnb, NodeContainer *obsUes)
{
    uint32_t nrObsUe = obsUes->GetN();

    Ptr<OpenGymBoxContainer<float>> bitrateDemandBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>({
        nrObsUe,
    }));
    for (uint32_t i = 0; i < nrObsUe; ++i)
    {
        uint32_t ueNodeId = obsUes->Get(i)->GetId();
        UeQosRequirementState state = GetUeQosRequirementState(ueNodeId);
        bitrateDemandBox->AddValue(state.requestedBitrate);
    }

    Ptr<OpenGymBoxContainer<float>> servCellIdBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>({
        nrObsUe,
    }));
    for (uint32_t i = 0; i < nrObsUe; ++i)
    {
        uint32_t ueNodeId = obsUes->Get(i)->GetId();
        uint16_t cellId = ueNodeIdToServingCellIds[ueNodeId];
        servCellIdBox->AddValue(cellId);
    }

    Ptr<OpenGymBoxContainer<float>> mcsBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>({
        nrObsUe,
    }));
    for (uint32_t i = 0; i < nrObsUe; ++i)
    {
        // uint32_t imsi = obsUes->Get (i)->GetDevice (1)->GetObject<LteUeNetDevice> ()->GetImsi ();
        uint32_t cellId = obsUes->Get(i)->GetDevice(1)->GetObject<LteUeNetDevice>()->GetRrc()->GetCellId();
        uint32_t rnti = obsUes->Get(i)->GetDevice(1)->GetObject<LteUeNetDevice>()->GetRrc()->GetRnti();
        // double mcs = ueImsiToMeanMcss[imsi];
        double mcs = cellIdToRntiToMeanMcss[cellId][rnti];
        mcsBox->AddValue(mcs);
    }

    Ptr<OpenGymBoxContainer<float>> rsrqBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>({
        nrObsUe,
        nrEnb,
    }));
    for (uint32_t i = 0; i < nrObsUe; ++i)
    {
        uint32_t ueNodeId = obsUes->Get(i)->GetId();
        for (uint32_t j = 0; j < nrEnb; ++j)
        {
            float rsrq = ueNodeIdToCellIdToRsrqs[ueNodeId][cellIds[j]];
            rsrqBox->AddValue(rsrq < -100 ? -100 : rsrq);
        }
    }

    Ptr<OpenGymBoxContainer<float>> rsrpBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>({
        nrObsUe,
        nrEnb,
    }));
    for (uint32_t i = 0; i < nrObsUe; ++i)
    {
        uint32_t ueNodeId = obsUes->Get(i)->GetId();
        for (uint32_t j = 0; j < nrEnb; ++j)
        {
            float rsrp = ueNodeIdToCellIdToRsrps[ueNodeId][cellIds[j]];
            rsrpBox->AddValue(rsrp < -100 ? -100 : rsrp);
        }
    }

    Ptr<OpenGymBoxContainer<float>> newDataRatioBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>({
        nrEnb,
    }));
    for (uint32_t i = 0; i < nrEnb; ++i)
    {
        uint32_t enbNodeId = enbNodeIds[i];
        float ndiPerNtx = 1.0;
        if (enbNodeIdToPerSecNrDlPhyTxs[enbNodeId] > 0)
        {
            ndiPerNtx = (float)enbNodeIdToPerSecNrDlPhyNdis[enbNodeId] / (float)enbNodeIdToPerSecNrDlPhyTxs[enbNodeId];
        }
        newDataRatioBox->AddValue(ndiPerNtx);
    }

    Ptr<OpenGymBoxContainer<float>> subBandOffsetBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>({
        nrEnb,
    }));
    for (uint32_t i = 0; i < nrEnb; ++i)
    {
        subBandOffsetBox->AddValue(cellSubBandOffsets[i]);
    }

    Ptr<OpenGymBoxContainer<float>> cellIdBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>({
        nrEnb,
    }));
    for (uint32_t i = 0; i < nrEnb; ++i)
    {
        cellIdBox->AddValue(cellIds[i]);
    }

    Ptr<OpenGymDiscreteContainer> hoUeIndexBox = CreateObject<OpenGymDiscreteContainer>();
    hoUeIndexBox->SetValue(currentHandoverUeIndex);

    uint32_t ueNodeId = obsUes->Get(currentHandoverUeIndex)->GetId();
    uint16_t servingCellId = ueNodeIdToServingCellIds[ueNodeId];
    Ptr<OpenGymDiscreteContainer> servCellIndexBox = CreateObject<OpenGymDiscreteContainer>();
    for (uint32_t i = 0; i < nrEnb; ++i)
    {
        if (cellIds[i] == servingCellId)
        {
            servCellIndexBox->SetValue(i);
        }
    }

    Ptr<OpenGymBoxContainer<float>> simTimeBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>({
        1,
    }));
    simTimeBox->AddValue(Simulator::Now().GetSeconds());

    Ptr<OpenGymBoxContainer<float>> qoeBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>({
        nrObsUe,
    }));
    for (uint32_t i = 0; i < nrObsUe; ++i)
    {
        Ptr<Node> ue = obsUes->Get(i);
        uint32_t ueNodeId = ue->GetId();
        double qoeScore = ueNodeIdToUeQoeFlowStates.at(ueNodeId).qoeScore;
        qoeBox->AddValue(qoeScore);
    }
    Ptr<OpenGymBoxContainer<float>> enbAverageDelayBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>({
        nrEnb,
    }));
    for (uint32_t i = 0; i < nrEnb; ++i)
    {
        double totalDelay = 0;
        double count = 0;
        for (uint32_t j = 0; j < nrObsUe; ++j)
        {
            Ptr<Node> ue = obsUes->Get(j);
            uint32_t ueNodeId = ue->GetId();
            uint16_t servingCellId = ueNodeIdToServingCellIds[ueNodeId];
            if (servingCellId == cellIds[i])
            {
                totalDelay += ueNodeIdToDelayPerSecond[ueNodeId];
                count++;
            }
        }
        double enbAverageDelay;
        if (count > 0)
        {
            enbAverageDelay = totalDelay / count;
        }
        else
        {
            enbAverageDelay = 0;
        }
        enbAverageDelayBox->AddValue(enbAverageDelay);
    }

    Ptr<OpenGymDictContainer> box = CreateObject<OpenGymDictContainer>();
    box->Add("bitrateDemand", bitrateDemandBox);
    box->Add("servCellId", servCellIdBox);
    box->Add("mcs", mcsBox);
    box->Add("rsrq", rsrqBox);
    box->Add("rsrp", rsrpBox);
    box->Add("newDataRatio", newDataRatioBox);
    box->Add("subBandOffset", subBandOffsetBox);
    box->Add("cellId", cellIdBox);
    box->Add("hoUeIndex", hoUeIndexBox);
    box->Add("servCellIndex", servCellIndexBox);
    box->Add("simTime", simTimeBox);
    box->Add("qoe", qoeBox);
    box->Add("enbDelay", enbAverageDelayBox);
    return box;
}

float GetG12Reward(NodeContainer *obsUes, uint32_t qoeType)
{
    float reward = 0;
    uint32_t ueNodeId = obsUes->Get(currentHandoverUeIndex)->GetId();
    // uint32_t qtype = obsUeInfo.at("qtype").at(ueNodeId - 28);
    uint32_t qtype = 4;
    // UpdateUeTriggerFlowStats (monitor, classifier);
    QosRequirement qosRequirement = GetUeQosRequirement(ueNodeId);
    double bitrate = NewUeTriggerOneSecAchievedBitrate(ueNodeId);
    double delay = NewUeTriggerOneSecAchievedDelay(ueNodeId);
    double errorRate = NewUeTriggerOneSecAchievedErrorRate(ueNodeId);
    double qoeScore = NewUeQoeScore(bitrate, delay, errorRate, qosRequirement, qoeType, qtype);
    reward = qoeScore;
    // std::cout << ueNodeId << std::endl;
    // fgetc(stdin);
    return reward;
}

bool GetG12GameOver()
{
    bool isGameOver = false;
    return isGameOver;
}

std::string
GetG12ExtraInfo()
{
    std::string info = "";
    return info;
}

bool ExeG12Acts(NodeContainer *obsUes, Ptr<OpenGymDataContainer> action)
{
    Ptr<Node> ue = obsUes->Get(currentHandoverUeIndex);
    uint32_t ueNodeId = ue->GetId();
    Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
    uint16_t targetCellId = cellIds[discrete->GetValue()];
    uint16_t servingCellId = ueNodeIdToServingCellIds[ueNodeId];
    if (servingCellId > 0 && servingCellId != targetCellId)
    {
        Ptr<NetDevice> device = ue->GetDevice(1);
        lte->HandoverRequest(Seconds(0), device, cellIdToPtrLteEnbNetDevices[servingCellId], targetCellId);
    }
    return true;
}

Ptr<OpenGymInterface>
CreateG12Iface(uint32_t openGymPort, uint32_t nrEnb, NodeContainer *obsUes, uint32_t qoeType)
{
    Ptr<OpenGymInterface> iface = CreateObject<OpenGymInterface>(openGymPort);
    iface->SetGetActionSpaceCb(MakeBoundCallback(&GetG12ActSpace, nrEnb));
    iface->SetGetObservationSpaceCb(MakeBoundCallback(&GetG12ObsSpace, nrEnb, obsUes));
    iface->SetGetObservationCb(MakeBoundCallback(&GetG12Obs, nrEnb, obsUes));
    iface->SetGetRewardCb(MakeBoundCallback(&GetG12Reward, obsUes, qoeType));
    iface->SetGetGameOverCb(MakeCallback(&GetG12GameOver));
    iface->SetGetExtraInfoCb(MakeCallback(&GetG12ExtraInfo));
    iface->SetExecuteActionsCb(MakeBoundCallback(&ExeG12Acts, obsUes));
    return iface;
}

Ptr<OpenGymInterface>
CreateOpenGymIface(std::string agentName, uint32_t openGymPort,
                   uint32_t nrEnb, NodeContainer *obsUes,
                   uint32_t qoeType)
{
    using fp_t = Ptr<OpenGymInterface> (*)(uint32_t, uint32_t, NodeContainer *, uint32_t);
    std::vector<std::pair<std::regex, fp_t>> fps = {
        {std::regex{"testG12"}, CreateG12Iface},
        {std::regex{"maxRsrqG12"}, CreateG12Iface},
        {std::regex{"minUeG12"}, CreateG12Iface},
        {std::regex{"randomG12"}, CreateG12Iface},
        {std::regex{"nnMaxQoeMarginG12[A-Za-z0-9]*"}, CreateG12Iface},
        {std::regex{"rlNnMaxQoeMarginG12[A-Za-z0-9]*"}, CreateG12Iface},
    };

    const auto it = std::find_if(fps.cbegin(), fps.cend(), [&](const std::pair<std::regex, fp_t> &p)
                                 { return std::regex_match(agentName, p.first); });
    Ptr<OpenGymInterface> iface = it->second(openGymPort, nrEnb, obsUes, qoeType);
    return iface;
}
void setUpBuildings()
{
    Ptr<GridBuildingAllocator> gridBuildingAllocator;
    gridBuildingAllocator = CreateObject<GridBuildingAllocator>();
    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(20));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(23));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(1947));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1927));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(22));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(17));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(1937));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1884));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(15));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(12));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(1990));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1925));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(25));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(28));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(2024));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1918));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(35));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(20));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(2061));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1902));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(45));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(20));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(1985));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1873));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(16));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(16));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(1992));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1821));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(26));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(17));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(2041));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1826));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(13));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(16));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(2089));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1848));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(58));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(67));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(1895));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1698));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(36));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(43));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(1989));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1709));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(28));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(47));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(2062));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1696));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(18));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(35));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(1981));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1635));
    gridBuildingAllocator->Create(1);

    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(1));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(41));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(18));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(0));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(30));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(1));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(10));
    gridBuildingAllocator->SetAttribute("MinX", DoubleValue(2028));
    gridBuildingAllocator->SetAttribute("MinY", DoubleValue(1642));
    gridBuildingAllocator->Create(1);
}
void delayPerSecond()
{
    Simulator::Schedule(MilliSeconds(1000), &delayPerSecond);
    std::cout << Simulator::Now().GetSeconds() << std::endl;
    for (const auto &delay : ueNodeIdToDelayPerSecond)
    {
        std::cout << "NodeId : " << delay.first << "delay : " << (delay.second * 10e-6) << "ms" << std::endl;
    }
    // fgetc(stdin);
}
void RxPDU(std::string context,
           uint16_t rnti,
           u_char lcid,
           uint32_t size,
           ulong delay)
{
    uint32_t nodeId = ConvertContextToNodeId(context);
    // std::cout << Simulator::Now().GetSeconds() << " NodeId: " << nodeId << " "
    //           << "delay: " << delay * 10e-6 << " ms" << std::endl;
    ueNodeIdToDelayPerSecond[nodeId] = delay;
}
void pdcpDelay()
{
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/LteUeRrc/DataRadioBearerMap/*/LtePdcp/RxPDU",
                    MakeCallback(&RxPDU));
}
int main(int argc, char *argv[])
{
    // parameter
    uint32_t seedNr = 1;
    uint64_t runNr = 0;
    uint32_t openGymPort = 5000;
    uint32_t simStopSec = 600;
    std::string bsInfoFn = "bsScen2Info7";
    std::string ctrlUeInfoFn = "ctrlUeScen2Info1";
    std::string obsUeInfoFn = "obsUeScen2Info1";
    std::string ctrlUeTraceFn = "ctrlUeScen2Poss1";
    std::string obsUeTraceFn = "obsUeScenTrace";
    std::string ctrlUeAttachFn = "ctrlUeScen2Attach1";
    std::string agentName = "none";
    uint32_t serialNr = 100;
    std::string resultDir = "/home/steven/ns-allinone-3.32/ns-3.32/Data/env13_result/";
    std::string infoDir = "/home/steven/ns-allinone-3.32/ns-3.32/Data/env7_info/";
    std::string simPrefix = "sim-env13-periodic-1-v1";
    uint32_t isEnableRlfDetection = 1;
    int32_t qOut = -8;
    uint32_t qoeType = 1;
    uint32_t triggerIntervalMilliSec = 25;
    uint32_t isEnableTrace = 0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("simSeed", "simSeed", runNr);
    cmd.AddValue("openGymPort", "openGymPort", openGymPort);
    cmd.AddValue("simStopSec", "simStopSec", simStopSec);
    cmd.AddValue("bsInfoFn", "bsInfoFn", bsInfoFn);
    cmd.AddValue("ctrlUeInfoFn", "ctrlUeInfoFn", ctrlUeInfoFn);
    cmd.AddValue("obsUeInfoFn", "obsUeInfoFn", obsUeInfoFn);
    cmd.AddValue("ctrlUeTraceFn", "ctrlUeTraceFn", ctrlUeTraceFn);
    cmd.AddValue("obsUeTraceFn", "obsUeTraceFn", obsUeTraceFn);
    cmd.AddValue("ctrlUeAttachFn", "ctrlUeAttachFn", ctrlUeAttachFn);
    cmd.AddValue("agentName", "agentName", agentName);
    cmd.AddValue("serialNr", "serialNr", serialNr);
    cmd.AddValue("resultDir", "resultDir", resultDir);
    cmd.AddValue("infoDir", "infoDir", infoDir);
    cmd.AddValue("simPrefix", "simPrefix", simPrefix);
    cmd.AddValue("isEnableRlfDetection", "isEnableRlfDetection", isEnableRlfDetection);
    cmd.AddValue("qOut", "qOut", qOut);
    cmd.AddValue("qoeType", "qoeType", qoeType);
    cmd.AddValue("triggerIntervalMilliSec", "triggerIntervalMilliSec", triggerIntervalMilliSec);
    cmd.AddValue("isEnableTrace", "isEnableTrace", isEnableTrace);
    cmd.Parse(argc, argv);

    std::stringstream simNameStream;
    simNameStream << simPrefix
                  << "_sss-" << simStopSec
                  << "_bif-" << bsInfoFn
                  << "_cuif-" << ctrlUeInfoFn
                  << "_ouif-" << obsUeInfoFn
                  << "_cutf-" << ctrlUeTraceFn
                  << "_outf-" << obsUeTraceFn
                  << "_cuaf-" << ctrlUeAttachFn
                  << "_ierd-" << isEnableRlfDetection
                  << "_qo-" << qOut
                  << "_qt-" << qoeType
                  << "_tims-" << triggerIntervalMilliSec
                  << "_an-" << agentName
                  << "_sn-" << serialNr;
    std::string simName = simNameStream.str();

    std::cout << simName << std::endl;
    std::cout << simName.size() << std::endl;

    // random
    RngSeedManager::SetSeed(seedNr);
    RngSeedManager::SetRun(runNr);
    std::srand(runNr);

    // config
    Config::SetDefault("ns3::LteUePhy::EnableRlfDetection", BooleanValue(isEnableRlfDetection));
    Config::SetDefault("ns3::LteUePhy::Qout", DoubleValue(qOut));
    Config::SetDefault("ns3::EpsBearer::Release", UintegerValue(15));
    Config::SetDefault("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue(320));
    Config::SetDefault("ns3::LteEnbMac::NumberOfRaPreambles", UintegerValue(10));
    Config::SetDefault("ns3::FlowMonitor::MaxPerHopDelay", TimeValue(Seconds(3600)));
    // Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (1024000));
    Config::SetDefault("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename", StringValue(resultDir + simName + "_DlRlcStats.txt"));
    Config::SetDefault("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename", StringValue(resultDir + simName + "_UlRlcStats.txt"));
    Config::SetDefault("ns3::RadioBearerStatsCalculator::DlPdcpOutputFilename", StringValue(resultDir + simName + "_DlPdcpStats.txt"));
    Config::SetDefault("ns3::RadioBearerStatsCalculator::UlPdcpOutputFilename", StringValue(resultDir + simName + "_UlPdcpStats.txt"));
    Config::SetDefault("ns3::MacStatsCalculator::DlOutputFilename", StringValue(resultDir + simName + "_DlMacStats.txt"));
    Config::SetDefault("ns3::MacStatsCalculator::UlOutputFilename", StringValue(resultDir + simName + "_UlMacStats.txt"));
    Config::SetDefault("ns3::PhyStatsCalculator::DlRsrpSinrFilename", StringValue(resultDir + simName + "_DlRsrpSinrStats.txt"));
    Config::SetDefault("ns3::PhyStatsCalculator::UlSinrFilename", StringValue(resultDir + simName + "_UlSinrStats.txt"));
    Config::SetDefault("ns3::PhyStatsCalculator::UlInterferenceFilename", StringValue(resultDir + simName + "UlInterferenceStats.txt"));
    Config::SetDefault("ns3::PhyTxStatsCalculator::DlTxOutputFilename", StringValue(resultDir + simName + "_DlTxPhyStats.txt"));
    Config::SetDefault("ns3::PhyTxStatsCalculator::UlTxOutputFilename", StringValue(resultDir + simName + "_UlTxPhyStats.txt"));
    Config::SetDefault("ns3::PhyRxStatsCalculator::DlRxOutputFilename", StringValue(resultDir + simName + "_DlRxPhyStats.txt"));
    Config::SetDefault("ns3::PhyRxStatsCalculator::UlRxOutputFilename", StringValue(resultDir + simName + "_UlRxPhyStats.txt"));
    Config::SetDefault("ns3::RadioBearerStatsCalculator::EpochDuration", TimeValue(Seconds(1)));

    setUpBuildings();

    // bsInfo, ueInfo
    std::map<std::string, std::vector<double>> enbInfo = GetEnbInfo(infoDir, bsInfoFn);
    uint32_t nrEnb = enbInfo.at("x").size();

    obsUeInfo = GetUeInfo(infoDir, obsUeInfoFn);
    uint32_t nrObsUe = obsUeInfo.at("qosRequirementChoice").size();
    uint32_t nrCtrlUe = 0;
    std::map<std::string, std::vector<double>> ctrlUeInfo;
    if (ctrlUeInfoFn != "none")
    {
        ctrlUeInfo = GetUeInfo(infoDir, ctrlUeInfoFn);
        nrCtrlUe = ctrlUeInfo.at("qosRequirementChoice").size();
    }
    std::cout << nrEnb << std::endl;
    std::cout << nrObsUe << std::endl;
    std::cout << nrCtrlUe << std::endl;
    std::cout << qosRequirementOptions.size() << std::endl;

    // node
    lte = CreateObject<LteHelper>();
    epc = CreateObject<PointToPointEpcHelper>();
    lte->SetEpcHelper(epc);

    Ptr<Node> remoteHost = CreateObject<Node>();
    Ptr<Node> pgw = epc->GetPgwNode();

    NodeContainer enbs;
    enbs.Create(nrEnb);
    SetEnbNodeIds(enbs);

    NodeContainer ues;

    NodeContainer obsUes;
    obsUes.Create(nrObsUe);
    ues.Add(obsUes);
    NodeContainer ctrlUes;
    if (nrCtrlUe > 0)
    {
        ctrlUes.Create(nrCtrlUe);
        ues.Add(ctrlUes);
    }
    lte->SetAttribute("PathlossModel", StringValue("ns3::HybridBuildingsPropagationLossModel"));
    lte->SetPathlossModelAttribute("ShadowSigmaExtWalls", DoubleValue(0));
    lte->SetPathlossModelAttribute("ShadowSigmaOutdoor", DoubleValue(1));
    lte->SetPathlossModelAttribute("ShadowSigmaIndoor", DoubleValue(1.5));
    // use always LOS model
    lte->SetPathlossModelAttribute("Los2NlosThr", DoubleValue(1e6));
    lte->SetSpectrumChannelType("ns3::MultiModelSpectrumChannel");
    // mobility
    Vector remoteHostPos = Vector(1800, 1600, 10);
    SetRemoteHostMobilityModel(remoteHost, remoteHostPos);

    Vector pgwPos = Vector(1810, 1600, 10);
    SetPgwMobilityModel(pgw, pgwPos);

    std::vector<Vector> enbPoss = GetEnbPoss(enbInfo);
    SetEnbsMobilityModel(enbs, enbPoss);

    if (nrCtrlUe > 0)
    {
        SetCtrlUesMobilityModel(ctrlUes, infoDir, ctrlUeTraceFn);
    }
    SetObsUesMobilityModel(obsUes, infoDir, obsUeTraceFn);

    // internet stack
    InternetStackHelper internetStack;
    internetStack.Install(remoteHost);
    internetStack.Install(ues);

    // internet
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2p.SetDeviceAttribute("Mtu", UintegerValue(1500));
    p2p.SetChannelAttribute("Delay", TimeValue(Seconds(0)));

    NetDeviceContainer internetDevices = p2p.Install(pgw, remoteHost);

    // address
    Ipv4AddressHelper ipv4Address;
    ipv4Address.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpv4Interfaces = ipv4Address.Assign(internetDevices);

    // routing
    Ipv4StaticRoutingHelper ipv4StaticRouting;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting;
    remoteHostStaticRouting = ipv4StaticRouting.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // lte
    NetDeviceContainer enbDevices = CreateEnbDevices(enbs, enbInfo);
    NetDeviceContainer ueDevices;
    NetDeviceContainer ctrlUeDevices;
    if (nrCtrlUe > 0)
    {
        ctrlUeDevices = CreateUeDevices(ctrlUes);
        ueDevices.Add(ctrlUeDevices);
    }
    NetDeviceContainer obsUeDevices = CreateUeDevices(obsUes);
    ueDevices.Add(obsUeDevices);

    // ue ipv4interface and static routing
    Ipv4InterfaceContainer ueInternetIpv4Interfaces;
    Ipv4InterfaceContainer ctrlUeInternetIpv4Interfaces;
    if (nrCtrlUe > 0)
    {
        ctrlUeInternetIpv4Interfaces = CreateUeInternetIpv4Interfaces(ctrlUes);
        ueInternetIpv4Interfaces.Add(ctrlUeInternetIpv4Interfaces);
    }
    Ipv4InterfaceContainer obsUeInternetIpv4Interfaces = CreateUeInternetIpv4Interfaces(obsUes);
    ueInternetIpv4Interfaces.Add(obsUeInternetIpv4Interfaces);
    SetUeStaticRouting(ues);

    // attach
    std::map<std::string, std::vector<double>> ctrlUeAttachInfo;
    if (nrCtrlUe > 0)
    {
        ctrlUeAttachInfo = GetCtrlUeAttachInfo(infoDir, ctrlUeAttachFn);
        for (uint32_t i = 0; i < ctrlUeAttachInfo.at("ueIndex").size(); ++i)
        {
            uint32_t ueIndex = ctrlUeAttachInfo.at("ueIndex").at(i);
            uint32_t bsIndex = ctrlUeAttachInfo.at("bsIndex").at(i);
            std::cout << i << "\t" << ueIndex << "\t" << bsIndex << "\t" << std::endl;
            lte->Attach(ctrlUeDevices.Get(ueIndex), enbDevices.Get(bsIndex));
        }
    }
    lte->Attach(obsUeDevices);

    // X2
    lte->AddX2Interface(enbs);

    // activate eps bearer
    if (nrCtrlUe > 0)
    {
        ActivateCtrlUeEpsBearers(ctrlUes, ctrlUeInfo);
    }
    ActivateObsUeEpsBearers(obsUes, obsUeInfo);

    // application
    ApplicationContainer serverApps = CreateServerApps(remoteHost, ues);
    ApplicationContainer clientApps;
    ApplicationContainer ctrlClientApps;
    if (nrCtrlUe > 0)
    {
        ctrlClientApps = CreateClientApps(remoteHost, ctrlUes, ctrlUeInternetIpv4Interfaces, ctrlUeInfo);
        clientApps.Add(ctrlClientApps);
    }
    ApplicationContainer obsClientApps = CreateClientApps(remoteHost, obsUes, obsUeInternetIpv4Interfaces, obsUeInfo);
    clientApps.Add(obsClientApps);
    serverApps.Start(Seconds(0.01));
    clientApps.Start(Seconds(0.5));
    Simulator::Schedule(MilliSeconds(1000), &pdcpDelay);
    Simulator::Schedule(MilliSeconds(2000), &delayPerSecond);

    // trace
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/LteUeRrc/ConnectionEstablished", MakeCallback(&LteUeRrcConnectionEstablishedCb));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/LteUeRrc/RadioLinkFailure", MakeCallback(&LteUeRrcRadioLinkFailureCb));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/LteUeRrc/HandoverEndOk", MakeCallback(&LteUeRrcHandoverEndOkCb));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/ComponentCarrierMapUe/*/LteUePhy/ReportUeMeasurements", MakeCallback(&LteUePhyReportUeMeasurementsCb));
    // std::ofstream ueMeasurementOfs;
    // ueMeasurementOfs.open(resultDir+simName+"_ueMeasurement.txt");
    // Config::Connect ("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/ComponentCarrierMapUe/*/LteUePhy/ReportUeMeasurements", MakeBoundCallback (&PrintLteUePhyReportUeMeasurementsCb, &ueMeasurementOfs));
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::UdpServer/RxWithAddresses", MakeCallback(&UdpServerRxWithAddressesCb));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/ComponentCarrierMapUe/*/LteUePhy/DlSpectrumPhy/RxEndOk", MakeCallback(&DlSpectrumPhyRxEndOkCb));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteUeNetDevice/ComponentCarrierMapUe/*/LteUePhy/DlSpectrumPhy/RxEndError", MakeCallback(&DlSpectrumPhyRxEndErrorCb));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteEnbNetDevice/ComponentCarrierMap/*/LteEnbPhy/DlSpectrumPhy/TxEnd", MakeCallback(DlSpectrumPhyTxEndCb));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteEnbNetDevice/ComponentCarrierMap/*/LteEnbMac/DlScheduling", MakeCallback(&LteEnbMacDlSchedulingCb));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::LteEnbNetDevice/ComponentCarrierMap/*/LteEnbPhy/DlPhyTransmission", MakeCallback(&DlPhyTransmissionCb));
    // Config::Connect ("/NodeList/*/DeviceList/*/$ns3::LteEnbNetDevice/ComponentCarrierMap/*/FfMacScheduler/$ns3::PfFfMacSchedulerV2/RntiMcs", MakeCallback (&RntiMcsCb));

    EnsureConnected();

    // initialize global variables
    InitUeNodeIdToServingCellIds(ues);
    if (nrCtrlUe > 0)
    {
        SetUeNodeIdToQosRequirementChoices(ctrlUes, ctrlUeInfo);
    }
    SetUeNodeIdToQosRequirementChoices(obsUes, obsUeInfo);
    SetCellIdToBwHzs(enbs, enbInfo);
    InitUeFlowStats(ues);

    // flowmonitor
    FlowMonitorHelper flowMonitor;
    flowMonitor.Install(remoteHost);
    flowMonitor.Install(ues);
    monitor = flowMonitor.GetMonitor();
    classifier = DynamicCast<Ipv4FlowClassifier>(flowMonitor.GetClassifier());
    Simulator::Schedule(Seconds(1), &SchedulePerSecUpdateUeFlowStats);
    // ue qoe score
    Simulator::Schedule(Seconds(1), &SchedulePerSecSetUeNodeIdToUeQoeFlowStates, ues, qoeType);
    // enb subframe usage
    Simulator::Schedule(Seconds(1), &SchedulePerSecSetEnbNodeIdToNrUsedSubframes, enbs);
    // enb dlphy ntx ndi
    Simulator::Schedule(Seconds(1), &SchedulePerSecSetEnbNodeIdToPerSecNrDlPhyTxNdis, enbs);
    // enb mcs dev distr
    // Simulator::Schedule (Seconds (1), &SchedulePerSecSetEnbNodeIdToMcsDevDistrs, enbs);
    // Simulator::Schedule (Seconds (1), &SchedulePerSecSetUeImsiToMeanMcss, ues);
    Simulator::Schedule(Seconds(1), &SchedulePerSecSetCellIdToRntiToMeanMcss);
    std::cout << "TT1" << std::endl;

    // opengym
    Ptr<OpenGymInterface> openGymIface = CreateOpenGymIface(agentName, openGymPort, nrEnb, &obsUes, qoeType);

    for (uint32_t i = 0; i < obsUes.GetN(); ++i)
    {
        Simulator::Schedule(Seconds(1 + 0.001 * triggerIntervalMilliSec * (i + 1)), &ScheduleObsUePeriodicTrigger, openGymIface, &obsUes, i, triggerIntervalMilliSec);
    }

    // output
    // std::ofstream enbInfoOfs;
    // enbInfoOfs.open(resultDir + simName + "_enbInfo.txt");
    // PrintEnbInfo(enbInfoOfs, enbs, enbInfo);

    // if (nrCtrlUe > 0)
    // {
    //     std::ofstream ctrlUeInfoOfs;
    //     ctrlUeInfoOfs.open(resultDir + simName + "_ctrlUeInfo.txt");
    //     PrintUeInfo(ctrlUeInfoOfs, ctrlUes, ctrlUeInfo);
    // }

    // std::ofstream obsUeInfoOfs;
    // obsUeInfoOfs.open(resultDir + simName + "_obsUeInfo.txt");
    // PrintUeInfo(obsUeInfoOfs, obsUes, obsUeInfo);

    // std::ofstream ueImsiOfs;
    // ueImsiOfs.open(resultDir + simName + "_ueImsi.txt");
    // PrintUeImsi(ueImsiOfs, ues);

    // schedule output
    // std::ofstream nodePossOfs;
    // nodePossOfs.open(resultDir + simName + "_obsUePos.txt");
    // Simulator::Schedule(Seconds(0), &SchedulePerSecPrintNodePoss, &nodePossOfs, obsUes);

    // std::ofstream eachCellNrServingUeOfs;
    // eachCellNrServingUeOfs.open(resultDir + simName + "_nrServingUe.txt");
    // Simulator::Schedule(Seconds(0), &SchedulePerSecPrintEachCellNrServingUe, &eachCellNrServingUeOfs);

    // std::ofstream ueRxBytesOfs;
    // ueRxBytesOfs.open(resultDir + simName + "_obsUeRxByte.txt");
    // Simulator::Schedule(Seconds(0), &SchedulePerSecPrintUeRxBytes, &ueRxBytesOfs, obsUes);

    // std::ofstream ueNrRlfsOfs;
    // ueNrRlfsOfs.open(resultDir + simName + "_obsUeNrRlf.txt");
    // Simulator::Schedule(Seconds(0), &SchedulePerSecPrintUeNrRlfs, &ueNrRlfsOfs, obsUes);

    // std::ofstream ueNrHandoversOfs;
    // ueNrHandoversOfs.open(resultDir + simName + "_obsUeNrHandover.txt");
    // Simulator::Schedule(Seconds(0), &SchedulePerSecPrintUeNrHandovers, &ueNrHandoversOfs, obsUes);

    // std::ofstream ueNrPingpongsOfs;
    // ueNrPingpongsOfs.open(resultDir + simName + "_obsUeNrPingpong.txt");
    // Simulator::Schedule(Seconds(0), &SchedulePerSecPrintUeNrPingpongs, &ueNrPingpongsOfs, obsUes);

    // std::ofstream ueServingCellIdsOfs;
    // ueServingCellIdsOfs.open(resultDir + simName + "_obsUeServingCellId.txt");
    // Simulator::Schedule(Seconds(0), &SchedulePerSecPrintUeServingCellIds, &ueServingCellIdsOfs, obsUes);

    // std::ofstream ueQoeFlowStatesOfs;
    // ueQoeFlowStatesOfs.open(resultDir + simName + "_obsUeQoeFlowState.txt");
    // Simulator::Schedule(Seconds(0), &SchedulePerSecPrintUeQoeFlowStates, &ueQoeFlowStatesOfs, obsUes);

    // std::ofstream per25msFlowStatsOfs;
    // per25msFlowStatsOfs.open (resultDir+simName+"_per25msFlowStats.txt");
    // Simulator::Schedule (Seconds (0), &SchedulePer25msPrintFlowStats, &per25msFlowStatsOfs);

    // std::ofstream per25msUeServingCellIdsOfs;
    // per25msUeServingCellIdsOfs.open (resultDir+simName+"_per25msObsUeServingCellId.txt");
    // Simulator::Schedule (Seconds (0), &SchedulePer25msPrintUeServingCellIds, &per25msUeServingCellIdsOfs, obsUes);

    // std::ofstream per25msUeServingCellIdRsrqRsrpSinrsOfs;
    // per25msUeServingCellIdRsrqRsrpSinrsOfs.open (resultDir+simName+"_per25msObsUeServingCellIdRsrqRsrpSinr.txt");
    // Simulator::Schedule (Seconds (0), &SchedulePer25msPrintUeServingCellIdRsrqRsrpSinrs, &per25msUeServingCellIdRsrqRsrpSinrsOfs, obsUes);

    // sim
    if (isEnableTrace)
    {
        std::cout << "Enable Traces" << std::endl;
        // lte->EnablePhyTraces();
        // lte->EnableMacTraces();
        // lte->EnableRlcTraces();
        lte->EnablePdcpTraces();
    }
    Ptr<RadioBearerStatsCalculator> pdcpStats = lte->GetPdcpStats();
    pdcpStats->SetAttribute("EpochDuration", TimeValue(Seconds(1)));
    Time simStopTime = Seconds(simStopSec);
    Simulator::Stop(simStopTime);
    Simulator::Run();

    openGymIface->NotifySimulationEnd();
    Simulator::Destroy();

    return 0;
}
