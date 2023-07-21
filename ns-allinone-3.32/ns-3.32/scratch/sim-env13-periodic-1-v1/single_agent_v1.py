
import os

from tensorflow.python.keras.engine import input_layer
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import os.path
import argparse
import glob
import collections
import re
import copy

import numpy as np
np.set_printoptions(precision=3, suppress=True)
import pandas as pd

from ns3gym import ns3env

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import layers
from tf_agents import specs
from tf_agents.replay_buffers import py_uniform_replay_buffer

def new_nr_bs(args):
    path = os.path.join(args.infoDir, f"{args.bsInfoFn}.tsv")
    nr_bs = 0
    with open(path) as f:
        for row in f:
            nr_bs += 1
    nr_bs -= 1
    return nr_bs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",
                        type=int,
                        default=6000,
                        help="6000, 6001, ...")
    parser.add_argument("--simStopSec",
                        type=int,
                        default=10,
                        help="3600, 10")
    parser.add_argument("--bsInfoFn",
                        type=str,
                        default="bsScen2Info7",
                        help="bsScen2Info7")
    parser.add_argument("--ctrlUeInfoFn",
                        type=str,
                        default="ctrlUeScen2Info1",
                        help="ctrlUeScen2Info1")
    parser.add_argument("--obsUeInfoFn",
                        type=str,
                        default="obsUeScen2Info1",
                        help="obsUeScen2Info1")
    parser.add_argument("--ctrlUeTraceFn",
                        type=str,
                        default="ctrlUeScen2Poss1",
                        help="ctrlUeScen2Trace1")
    parser.add_argument("--obsUeTraceFn",
                        type=str,
                        default="obsUeScen2Poss2",
                        help="obsUeScen2Trace1")
    parser.add_argument("--ctrlUeAttachFn",
                        type=str,
                        default="ctrlUeScen2Attach1",
                        help="ctrlUeScen2Attach1")
    parser.add_argument("--agentName",
                        type=str,
                        default="None",
                        help="maxRsrqG6, minUeG6, nnMaxQoeMarginG6Rl1")
    parser.add_argument("--isEnableRlfDetection",
                        type=int,
                        default=1,
                        help="1, 0")
    parser.add_argument("--qOut",
                        type=int,
                        default=-8,
                        help="-5, -8")
    parser.add_argument("--qoeType",
                        type=int,
                        default=1,
                        help="0, 1,")
    parser.add_argument("--triggerIntervalMilliSec",
                        type=int,
                        default=25,
                        help="25")
    parser.add_argument("--isEnableTrace",
                        type=int,
                        default=0,
                        help="0, 1")
    parser.add_argument('--iterations',
                    type=int,
                    default=2,
                    help='Number of iterations, Default: 2')
    args = parser.parse_args()
    args.resultDir = str("/home/steven/ns-allinone-3.32/ns-3.32/Data/env13_result/")
    args.infoDir = str("/home/steven/ns-allinone-3.32/ns-3.32/Data/env7_info/")
    args.simPrefix = str("sim-env13-periodic-1-v1")
    args.serialNr = get_serialNr(args)
    args.simSeed = 1
    args.agentSeed = 1
    args.debug = True
    args.nrBs = new_nr_bs(args)
    return args

def get_serialNr(args):
    serialNr = 0
    simName = (
        f"{args.simPrefix}"
        f"_sss-{args.simStopSec}"
        f"_bif-{args.bsInfoFn}"
        f"_cuif-{args.ctrlUeInfoFn}"
        f"_ouif-{args.obsUeInfoFn}"
        f"_cutf-{args.ctrlUeTraceFn}"
        f"_outf-{args.obsUeTraceFn}"
        f"_cuaf-{args.ctrlUeAttachFn}"
        f"_ierd-{args.isEnableRlfDetection}"
        f"_qo-{args.qOut}"
        f"_qt-{args.qoeType}"
        f"_tims-{args.triggerIntervalMilliSec}"
        f"_an-{args.agentName}"
        f"_sn-{serialNr}"
    )
    path = os.path.join(args.resultDir, f"{simName}*")
    existings = glob.glob(path)
    while existings:
        serialNr += 1
        simName = (
            f"{args.simPrefix}"
            f"_sss-{args.simStopSec}"
            f"_bif-{args.bsInfoFn}"
            f"_cuif-{args.ctrlUeInfoFn}"
            f"_ouif-{args.obsUeInfoFn}"
            f"_cutf-{args.ctrlUeTraceFn}"
            f"_outf-{args.obsUeTraceFn}"
            f"_cuaf-{args.ctrlUeAttachFn}"
            f"_ierd-{args.isEnableRlfDetection}"
            f"_qo-{args.qOut}"
            f"_qt-{args.qoeType}"
            f"_tims-{args.triggerIntervalMilliSec}"
            f"_an-{args.agentName}"
            f"_sn-{serialNr}"
        )
        path = os.path.join(args.resultDir, f"{simName}*")
        existings = glob.glob(path)
    return serialNr

def get_env(args):
    sim_args = {
        "--simStopSec": args.simStopSec,
        "--bsInfoFn": args.bsInfoFn,
        "--ctrlUeInfoFn": args.ctrlUeInfoFn,
        "--obsUeInfoFn": args.obsUeInfoFn,
        "--ctrlUeTraceFn": args.ctrlUeTraceFn,
        "--obsUeTraceFn": args.obsUeTraceFn,
        "--ctrlUeAttachFn": args.ctrlUeAttachFn,
        "--agentName": args.agentName,
        "--serialNr": args.serialNr,
        "--resultDir": args.resultDir,
        "--infoDir": args.infoDir,
        "--simPrefix": args.simPrefix,
        "--isEnableRlfDetection": args.isEnableRlfDetection,
        "--qOut": args.qOut,
        "--qoeType": args.qoeType,
        "--triggerIntervalMilliSec": args.triggerIntervalMilliSec,
        "--isEnableTrace": args.isEnableTrace,
    }
    env = ns3env.Ns3Env(port=args.port, simSeed=args.simSeed, 
                        simArgs=sim_args, debug=args.debug)
    return env

def get_simName(args):
    simName = (
        f"{args.simPrefix}"
        f"_sss-{args.simStopSec}"
        f"_bif-{args.bsInfoFn}"
        f"_cuif-{args.ctrlUeInfoFn}"
        f"_ouif-{args.obsUeInfoFn}"
        f"_cutf-{args.ctrlUeTraceFn}"
        f"_outf-{args.obsUeTraceFn}"
        f"_cuaf-{args.ctrlUeAttachFn}"
        f"_ierd-{args.isEnableRlfDetection}"
        f"_qo-{args.qOut}"
        f"_qt-{args.qoeType}"
        f"_tims-{args.triggerIntervalMilliSec}"
        f"_an-{args.agentName}"
        f"_sn-{args.serialNr}"
    )
    return simName

def load_data_frame(args, suffix, **kwargs):
    simName = get_simName(args)
    path = os.path.join(args.resultDir, f"{simName}_{suffix}")
    df = pd.read_csv(path, sep="\s+", **kwargs)
    return df

def load_obsUeObsReward_df(args):
    df = load_data_frame(args, "obsUeObsReward.txt", header=0)
    return df

class BaseG12:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.columns = self.new_columns()
        self.PrevObs = self.new_PrevObs()

    def new_columns(self):
        columns = (
            "simTime", "ueIndex", "cellIndex", "bitrateDemand", "rsrq", "newDataRatio",
        ) + tuple(
            f"nrUeForDemandType{i}" for i in range(5)
        ) + ("mcs", "qoeScore", )
        return columns

    def new_PrevObs(self):
        PrevObs = collections.namedtuple("PrevObs", self.columns[:-1])
        return PrevObs

    def open_file(self):
        args = self.args
        simName = get_simName(args)
        path = os.path.join(args.resultDir, f"{simName}_obsUeObsReward.txt")
        f = open(path, "w")
        return f

    def write_columns(self, f):
        for column in self.columns:
            f.write(f"{column}\t")
        f.write("\n")

    def write_prev_obs_reward(self, f, prev_obs, reward):
        f.write(f"{prev_obs.simTime:.3f}\t")
        f.write(f"{prev_obs.ueIndex}\t")
        f.write(f"{prev_obs.cellIndex}\t")
        f.write(f"{prev_obs.bitrateDemand}\t")
        f.write(f"{prev_obs.rsrq:.3f}\t")
        f.write(f"{prev_obs.newDataRatio:.3f}\t")
        for i in range(0, 5):
            attr = f"nrUeForDemandType{i}"
            f.write(f"{getattr(prev_obs, attr)}\t")
        f.write(f"{prev_obs.mcs:.3f}\t")
        f.write(f"{reward:.6f}\t")
        f.write("\n")

    def create_prev_obs(self, obs, action):
        sim_time = obs["simTime"][0]
        ue_index = obs["hoUeIndex"]
        bs_index = action
        bitrate_demand = obs["bitrateDemand"][ue_index]
        rsrq = obs["rsrq"][ue_index, bs_index]
        new_data_ratio = obs["newDataRatio"][bs_index]

        cell_id = obs["cellId"][bs_index]
        other_indexs = np.where(obs["servCellId"]==cell_id)[0]
        demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
        nr_ues = np.zeros(5)
        for other_index in other_indexs:
            nr_ues[demand_indexs[obs["bitrateDemand"][other_index]]] += 1

        mcs = obs["mcs"][ue_index]

        if bs_index != obs["servCellIndex"]:
            nr_ues[demand_indexs[obs["bitrateDemand"][ue_index]]] += 1
            rsrps = obs["rsrp"][other_indexs, bs_index]
            if len(rsrps) == 0:
                mcs = 30
            else:
                rsrp = obs["rsrp"][ue_index, bs_index]
                closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                mcss = obs["mcs"][other_indexs]
                mcs = np.mean(mcss[closest_mcs_indexs])
        prev_obs = self.PrevObs(
            sim_time, ue_index, bs_index, bitrate_demand, rsrq, new_data_ratio, *nr_ues, mcs
        )
        return prev_obs

    def new_candidate_bs_indexs(self, obs):
        ue_index = obs["hoUeIndex"]
        indexs = np.where(obs["rsrq"][ue_index]>-12)[0]
        return indexs

class NoLearnBaseG12(BaseG12):
    def run(self):
        env = self.env
        prev_obss = {}

        with self.open_file() as f:
            self.write_columns(f)

            obs, reward, done, info = env.get_state()
            
            print(obs, reward, done, info)
            while not done:
                print("------------new action----------------")
                ue_index = obs["hoUeIndex"]
                print("-----------------------ue_index",ue_index,'------------------------')
                if ue_index in prev_obss:
                    prev_obs = prev_obss[ue_index]
                    self.write_prev_obs_reward(f, prev_obs, reward)

                candidate_bs_indexs = self.new_candidate_bs_indexs(obs)
                if len(candidate_bs_indexs) == 0:
                    action = obs["servCellIndex"]
                elif len(candidate_bs_indexs) == 1:
                    action = candidate_bs_indexs[0]
                else:
                    action = self.new_action(obs, candidate_bs_indexs)

                print("action:", action, "rsrq:", obs["rsrq"][ue_index][action])
                print(self.args.agentName)
                print(self.args.bsInfoFn)
                print(self.args.obsUeInfoFn)
                print(self.args.obsUeTraceFn)

                prev_obss[ue_index] = self.create_prev_obs(obs, action)

                obs, reward, done, info = env.step(action)
                print(obs, reward, done, info)

class MaxRsrqG12(NoLearnBaseG12):
    def new_action(self, obs, candidate_bs_indexs):
        ue_index = obs["hoUeIndex"]
        rsrqs = obs["rsrq"][ue_index]
        action = candidate_bs_indexs[np.argmax(rsrqs[candidate_bs_indexs])]
        return action

class MinUeG12(NoLearnBaseG12):
    def new_action(self, obs, candidate_bs_indexs):
        ue_index = obs["hoUeIndex"]
        #uniques, counts = np.unique(obs["servCellId"], return_counts=True)
        counts = collections.Counter(obs["servCellId"])
        counts[obs["cellId"][obs["servCellIndex"]]] -= 1
        nr_ues = np.array([counts[obs["cellId"][bs_index]] for bs_index in candidate_bs_indexs])
        #nr_ues = obs["nrUe"] + 1
        #nr_ues[obs["servCellIndex"]] -= 1
        action = candidate_bs_indexs[np.argmin(nr_ues)]
        return action

class RandomG12(NoLearnBaseG12):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.rng = np.random.default_rng(self.args.serialNr)

    def new_action(self, obs, candidate_bs_indexs):
        action = self.rng.choice(candidate_bs_indexs)
        return action

class QoeEventTriggerNoLearnBaseG12(NoLearnBaseG12):
    def run(self):
        env = self.env
        prev_obss = {}

        with self.open_file() as f:
            self.write_columns(f)

            obs, reward, done, info = env.get_state()
            print(obs, reward, done, info)
            while not done:
                ue_index = obs["hoUeIndex"]

                if ue_index in prev_obss:
                    prev_obs = prev_obss[ue_index]
                    self.write_prev_obs_reward(f, prev_obs, reward)

                candidate_bs_indexs = self.new_candidate_bs_indexs(obs)
                if len(candidate_bs_indexs) == 0:
                    action = obs["servCellIndex"]
                elif obs["qoe"][ue_index] >= self.qoe_threshold:
                    print("event is not triggered", obs["qoe"][ue_index], self.qoe_threshold)
                    action = obs["servCellIndex"]
                elif len(candidate_bs_indexs) == 1:
                    action = candidate_bs_indexs[0]
                else:
                    action = self.new_action(obs, candidate_bs_indexs)

                print("action:", action, "rsrq:", obs["rsrq"][ue_index][action])
                print(self.args.agentName)
                print(self.args.bsInfoFn)
                print(self.args.obsUeInfoFn)
                print(self.args.obsUeTraceFn)

                prev_obss[ue_index] = self.create_prev_obs(obs, action)

                obs, reward, done, info = env.step(action)
                print(obs, reward, done, info)




class RlNnMaxQoeMarginBaseG12(BaseG12):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.rng = np.random.default_rng(self.args.serialNr)
    
    def get_model_name(self):
        p = re.compile(r"RlNnMaxQoeMarginG12(Model\d+)")
        name = p.match(self.__class__.__name__).groups()[0]
        return name

    def new_model(self, bs_index):
        model = keras.Sequential([
            layers.Dense(100, activation="relu", input_shape=(self.input_shape, )),
            layers.Dense(100, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(loss="mean_absolute_error",
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def new_models(self, nr_bs):
        models = []
        for i in range(nr_bs):
            model = self.new_model(i)
            models.append(model)
        return models

    def new_buffer(self, data_spec, capacity):
        buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(data_spec, capacity)
        return buffer

    def new_buffers(self, nr_bs):
        capacity = 90000
        buffers = []
        for i in range(nr_bs):
            data_spec = self.new_data_spec(i)
            buffer = self.new_buffer(data_spec, capacity)
            buffers.append(buffer)
        return buffers

    def new_data_iters(self, buffers):
        data_iters = []
        for buffer in buffers:
            dataset = buffer.as_dataset(sample_batch_size=5120)
            data_iter = iter(dataset)
            data_iters.append(data_iter)
        return data_iters

    def train_model(self, models, buffers, data_iters, bs_index):
        if buffers[bs_index].size <= 0:
            return
        model = models[bs_index]
        data_iter = data_iters[bs_index]
        train_data = next(data_iter)
        print(train_data[0], train_data[1])
        #history = model.fit(train_data[0], train_data[1], epochs=50)
        history = model.fit(train_data[0], train_data[1])

    def train_models(self, models, buffers, data_iters, nr_bs):
        for bs_index in range(nr_bs):
            print(bs_index)
            self.train_model(models, buffers, data_iters, bs_index)

    def save_bs_model_weights(self, models, bs_index):
        #dn = "./checkpoints"
        #args = self.args
        #dn = f"./checkpoints/{args.agentName}"
        #dn = f"./checkpoints/{self.get_model_name()}"
        dn = f"./checkpoints/{self.get_model_name()}_{self.args.serialNr}"
        fn = f"bs{bs_index}_checkpoint"
        path = os.path.join(dn, fn)
        models[bs_index].save_weights(path)

    def get_obsUeObsReward_df(self, argss):
        dfs = []
        for args in argss:
            df = load_obsUeObsReward_df(args)
            dfs.append(df)
        result_df = pd.concat(dfs, axis=0, ignore_index=True)
        return result_df

    #def get_training_df(self):
    #    argss = self.new_training_argss()
    #    if len(argss) == 0:
    #        return None
    #    df = self.get_obsUeObsReward_df(argss)
    #    return df

    #def add_init_data_to_buffers(self, buffers, nr_bs):
    #    training_df = self.get_training_df()
    #    if training_df is None:
    #        return
    #    for bs_index in range(nr_bs):
    #        features, labels = self.get_bs_features_and_labels(training_df, bs_index)
    #        buffer = buffers[bs_index]
    #        for i in range(len(features)):
    #            label = np.zeros(1) + labels[i]
    #            item = (np.expand_dims(features[i],0), np.expand_dims(label, 0))
    #            buffer.add_batch(item)

    #def new_training_argss(self):
    #    argss = []
    #    sn = self.args.serialNr
    #    for i in range(sn):
    #        args = copy.deepcopy(self.args)
    #        args.serialNr = i
    #        argss.append(args)
    #    return argss

    def predict_serv_bs_qoe(self, models, obs, has_ho):
        bs_index = obs["servCellIndex"]
        cell_id = obs["cellId"][bs_index]
        ue_indexs = np.where(obs["servCellId"]==cell_id)[0]
        ho_ue_index = obs["hoUeIndex"]
        if has_ho:
            ue_indexs = ue_indexs[ue_indexs != ho_ue_index]
        bs_qoe = self.predict_bs_qoe(models, obs, bs_index, ue_indexs)
        return bs_qoe

    def predict_target_bs_qoe(self, models, obs, bs_index, has_ho):
        cell_id = obs["cellId"][bs_index]
        ue_indexs = np.where(obs["servCellId"]==cell_id)[0]
        ho_ue_index = obs["hoUeIndex"]
        if has_ho:
            ue_indexs = ue_indexs[ue_indexs != ho_ue_index]
            ue_indexs = np.append(ue_indexs, ho_ue_index)
        bs_qoe = self.predict_bs_qoe(models, obs, bs_index, ue_indexs)
        return bs_qoe

    def new_action(self, models, obs, candidate_bs_indexs):
        t = obs["simTime"][0]
        epsilon = 2**(-t/60)
        print(t, epsilon)
        if self.rng.random() < epsilon and t <= 300:
        # if True:
            print("random choice")
            action = self.rng.choice(candidate_bs_indexs)
            return action

        serv_cell_index = obs["servCellIndex"]
        serv_bs_qoe_before = self.predict_serv_bs_qoe(models, obs, False)
        serv_bs_qoe_after = self.predict_serv_bs_qoe(models, obs, True)
        margins = []
        for bs_index in candidate_bs_indexs:
            if bs_index == serv_cell_index:
                margin = 0
                margins.append(margin)
                continue
            target_bs_qoe_before = self.predict_target_bs_qoe(models, obs, bs_index, False)
            target_bs_qoe_after = self.predict_target_bs_qoe(models, obs, bs_index, True)
            margin = (target_bs_qoe_after + serv_bs_qoe_after) - (target_bs_qoe_before + serv_bs_qoe_before)
            margins.append (margin)
        margin_array = np.array(margins)
        action = candidate_bs_indexs[np.argmax(margin_array)]
        return action

    def run(self):
        env = self.env
        nr_bs = self.args.nrBs
        prev_obss = {}

        models = self.new_models(nr_bs)
        buffers = self.new_buffers(nr_bs)
        data_iters = self.new_data_iters(buffers)
        #self.add_init_data_to_buffers(buffers, nr_bs)

        with self.open_file() as f:
            self.write_columns(f)

            obs, reward, done, info = env.get_state()
            print(obs, reward, done, info)
            while not done:
                ue_index = obs["hoUeIndex"]

                if ue_index in prev_obss:
                    prev_obs = prev_obss[ue_index]
                    self.write_prev_obs_reward(f, prev_obs, reward)
                    # add batch to buffer and train model
                    self.add_batch_to_buffers(buffers, prev_obs, reward)
                    self.train_model(models, buffers, data_iters, prev_obs.cellIndex)

                candidate_bs_indexs = self.new_candidate_bs_indexs(obs)
                if len(candidate_bs_indexs) == 0:
                    action = obs["servCellIndex"]
                elif len(candidate_bs_indexs) == 1:
                    action = candidate_bs_indexs[0]
                else:
                    action = self.new_action(models, obs, candidate_bs_indexs)

                print("action:", action, "rsrq:", obs["rsrq"][ue_index][action])
                print(self.args.agentName)
                print(self.args.bsInfoFn)
                print(self.args.obsUeInfoFn)
                print(self.args.obsUeTraceFn)

                prev_obss[ue_index] = self.create_prev_obs(obs, action)

                obs, reward, done, info = env.step(action)
                print(obs, reward, done, info)

        # save model weight
        for i in range(nr_bs):
            self.save_bs_model_weights(models, i)

class RlAllModelG12(RlNnMaxQoeMarginBaseG12):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 9

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / 1000000
            rsrq = rsrq / 12
            new_data_ratio = new_data_ratio / 1
            nr_ues = nr_ues / 10
            mcs = mcs / 28
            feature_vec = np.array([bitrate_demand, rsrq, new_data_ratio, *nr_ues, mcs])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        feature_array = np.array([
            prev_obs.bitrateDemand / 1000000,
            prev_obs.rsrq / 12,
            prev_obs.newDataRatio / 1,
            *[getattr(prev_obs, f"nrUeForDemandType{i}") / 10 for i in range(5)],
            prev_obs.mcs / 28,
        ])
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / 1000000 
        f2 = df2[["rsrq"]].to_numpy() / 12 
        f3 = df2[["newDataRatio"]].to_numpy() / 1
        f4 = df2[[f"nrUeForDemandType{i}" for i in range(0,5)]].to_numpy() / 10
        f5 = df2[["mcs"]].to_numpy() / 28
        features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlNoDemandModelG12(RlNnMaxQoeMarginBaseG12):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 8

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            #bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            #bitrate_demand = bitrate_demand / 1000000
            rsrq = rsrq / 12
            new_data_ratio = new_data_ratio / 1
            nr_ues = nr_ues / 10
            mcs = mcs / 28
            feature_vec = np.array([
                #bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                *nr_ues, 
                mcs
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        feature_array = np.array([
            #prev_obs.bitrateDemand / 1000000,
            prev_obs.rsrq / 12,
            prev_obs.newDataRatio / 1,
            *[getattr(prev_obs, f"nrUeForDemandType{i}") / 10 for i in range(5)],
            prev_obs.mcs / 28,
        ])
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        #f1 = df2[["bitrateDemand"]].to_numpy() / 1000000 
        f2 = df2[["rsrq"]].to_numpy() / 12 
        f3 = df2[["newDataRatio"]].to_numpy() / 1
        f4 = df2[[f"nrUeForDemandType{i}" for i in range(0,5)]].to_numpy() / 10
        f5 = df2[["mcs"]].to_numpy() / 28
        #features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        features = np.concatenate([f2,f3,f4,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlNoRsrqModelG12(RlNnMaxQoeMarginBaseG12):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 8

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            #rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / 1000000
            #rsrq = rsrq / 12
            new_data_ratio = new_data_ratio / 1
            nr_ues = nr_ues / 10
            mcs = mcs / 28
            feature_vec = np.array([
                bitrate_demand, 
                #rsrq, 
                new_data_ratio, 
                *nr_ues, 
                mcs
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        feature_array = np.array([
            prev_obs.bitrateDemand / 1000000,
            #prev_obs.rsrq / 12,
            prev_obs.newDataRatio / 1,
            *[getattr(prev_obs, f"nrUeForDemandType{i}") / 10 for i in range(5)],
            prev_obs.mcs / 28,
        ])
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / 1000000 
        #f2 = df2[["rsrq"]].to_numpy() / 12 
        f3 = df2[["newDataRatio"]].to_numpy() / 1
        f4 = df2[[f"nrUeForDemandType{i}" for i in range(0,5)]].to_numpy() / 10
        f5 = df2[["mcs"]].to_numpy() / 28
        #features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        features = np.concatenate([f1,f3,f4,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels




class RlNoNewRatioModelG12(RlNnMaxQoeMarginBaseG12):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 8

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            #new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / 1000000
            rsrq = rsrq / 12
            #new_data_ratio = new_data_ratio / 1
            nr_ues = nr_ues / 10
            mcs = mcs / 28
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                #new_data_ratio, 
                *nr_ues, 
                mcs
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        feature_array = np.array([
            prev_obs.bitrateDemand / 1000000,
            prev_obs.rsrq / 12,
            #prev_obs.newDataRatio / 1,
            *[getattr(prev_obs, f"nrUeForDemandType{i}") / 10 for i in range(5)],
            prev_obs.mcs / 28,
        ])
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / 1000000 
        f2 = df2[["rsrq"]].to_numpy() / 12 
        #f3 = df2[["newDataRatio"]].to_numpy() / 1
        f4 = df2[[f"nrUeForDemandType{i}" for i in range(0,5)]].to_numpy() / 10
        f5 = df2[["mcs"]].to_numpy() / 28
        #features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        features = np.concatenate([f1,f2,f4,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlNoNrUeModelG12(RlNnMaxQoeMarginBaseG12):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 4

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            #demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            #nr_ues = np.zeros(5)
            #for some_ue_index in ue_indexs:
            #    nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / 1000000
            rsrq = rsrq / 12
            new_data_ratio = new_data_ratio / 1
            #nr_ues = nr_ues / 10
            mcs = mcs / 28
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                #*nr_ues, 
                mcs
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        feature_array = np.array([
            prev_obs.bitrateDemand / 1000000,
            prev_obs.rsrq / 12,
            prev_obs.newDataRatio / 1,
            #*[getattr(prev_obs, f"nrUeForDemandType{i}") / 10 for i in range(5)],
            prev_obs.mcs / 28,
        ])
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / 1000000 
        f2 = df2[["rsrq"]].to_numpy() / 12 
        f3 = df2[["newDataRatio"]].to_numpy() / 1
        #f4 = df2[[f"nrUeForDemandType{i}" for i in range(0,5)]].to_numpy() / 10
        f5 = df2[["mcs"]].to_numpy() / 28
        #features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        features = np.concatenate([f1,f2,f3,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels


class RlNoMcsModelG12(RlNnMaxQoeMarginBaseG12):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 8

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            #if ue_index != ho_ue_index:
            #    mcs = obs["mcs"][ue_index]
            #elif bs_index == obs["servCellIndex"]:
            #    mcs = obs["mcs"][ue_index]
            #else:
            #    other_indexs = ue_indexs[ue_indexs != ho_ue_index]
            #    rsrps = obs["rsrp"][other_indexs, bs_index]
            #    if len(rsrps) == 0:
            #        mcs = 30
            #    else:
            #        rsrp = obs["rsrp"][ho_ue_index, bs_index]
            #        closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
            #        mcss = obs["mcs"][other_indexs]
            #        mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / 1000000
            rsrq = rsrq / 12
            new_data_ratio = new_data_ratio / 1
            nr_ues = nr_ues / 10
            #mcs = mcs / 28
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                *nr_ues, 
                #mcs
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        #feature_shape = (9, )
        feature_shape = (8, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        feature_array = np.array([
            prev_obs.bitrateDemand / 1000000,
            prev_obs.rsrq / 12,
            prev_obs.newDataRatio / 1,
            *[getattr(prev_obs, f"nrUeForDemandType{i}") / 10 for i in range(5)],
            #prev_obs.mcs / 28,
        ])
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / 1000000 
        f2 = df2[["rsrq"]].to_numpy() / 12 
        f3 = df2[["newDataRatio"]].to_numpy() / 1
        f4 = df2[[f"nrUeForDemandType{i}" for i in range(0,5)]].to_numpy() / 10
        #f5 = df2[["mcs"]].to_numpy() / 28
        #features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        features = np.concatenate([f1,f2,f3,f4], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlNnMaxQoeMarginBaseG12SaveModel(BaseG12):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.rng = np.random.default_rng(self.args.serialNr)
        self.time_step = 0
        self.save_period = 4000
    
    def get_model_name(self):
        p = re.compile(r"RlNnMaxQoeMarginG12(Model\d+)")
        name = p.match(self.__class__.__name__).groups()[0]
        return name

    def new_model(self, bs_index):
        model = keras.Sequential([
            layers.Dense(100, activation="relu", input_shape=(self.input_shape, )),
            layers.Dense(100, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(loss="mean_absolute_error",
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def new_models(self, nr_bs):
        models = []
        for i in range(nr_bs):
            model = self.new_model(i)
            models.append(model)
        return models

    def new_buffer(self, data_spec, capacity):
        buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(data_spec, capacity)
        return buffer

    def new_buffers(self, nr_bs):
        capacity = 90000
        buffers = []
        for i in range(nr_bs):
            data_spec = self.new_data_spec(i)
            buffer = self.new_buffer(data_spec, capacity)
            buffers.append(buffer)
        return buffers

    def new_data_iters(self, buffers):
        data_iters = []
        for buffer in buffers:
            dataset = buffer.as_dataset(sample_batch_size=5120)
            data_iter = iter(dataset)
            data_iters.append(data_iter)
        return data_iters

    def train_model(self, models, buffers, data_iters, bs_index):
        if buffers[bs_index].size <= 0:
            return
        model = models[bs_index]
        data_iter = data_iters[bs_index]
        train_data = next(data_iter)
        print(train_data[0], train_data[1])
        #history = model.fit(train_data[0], train_data[1], epochs=50)
        history = model.fit(train_data[0], train_data[1])

    def train_models(self, models, buffers, data_iters, nr_bs):
        for bs_index in range(nr_bs):
            print(bs_index)
            self.train_model(models, buffers, data_iters, bs_index)

    def save_bs_model_weights(self, models, bs_index):
        #dn = "./checkpoints"
        #args = self.args
        #dn = f"./checkpoints/{args.agentName}"
        #dn = f"./checkpoints/{self.get_model_name()}"
        dn = f"./checkpoints/{self.get_model_name()}_{self.args.serialNr}_{self.time_step}"
        fn = f"bs{bs_index}_checkpoint"
        path = os.path.join(dn, fn)
        models[bs_index].save_weights(path)

    def get_obsUeObsReward_df(self, argss):
        dfs = []
        for args in argss:
            df = load_obsUeObsReward_df(args)
            dfs.append(df)
        result_df = pd.concat(dfs, axis=0, ignore_index=True)
        return result_df

    #def get_training_df(self):
    #    argss = self.new_training_argss()
    #    if len(argss) == 0:
    #        return None
    #    df = self.get_obsUeObsReward_df(argss)
    #    return df

    #def add_init_data_to_buffers(self, buffers, nr_bs):
    #    training_df = self.get_training_df()
    #    if training_df is None:
    #        return
    #    for bs_index in range(nr_bs):
    #        features, labels = self.get_bs_features_and_labels(training_df, bs_index)
    #        buffer = buffers[bs_index]
    #        for i in range(len(features)):
    #            label = np.zeros(1) + labels[i]
    #            item = (np.expand_dims(features[i],0), np.expand_dims(label, 0))
    #            buffer.add_batch(item)

    #def new_training_argss(self):
    #    argss = []
    #    sn = self.args.serialNr
    #    for i in range(sn):
    #        args = copy.deepcopy(self.args)
    #        args.serialNr = i
    #        argss.append(args)
    #    return argss

    def predict_serv_bs_qoe(self, models, obs, has_ho):
        bs_index = obs["servCellIndex"]
        cell_id = obs["cellId"][bs_index]
        ue_indexs = np.where(obs["servCellId"]==cell_id)[0]
        ho_ue_index = obs["hoUeIndex"]
        if has_ho:
            ue_indexs = ue_indexs[ue_indexs != ho_ue_index]
        bs_qoe = self.predict_bs_qoe(models, obs, bs_index, ue_indexs)
        return bs_qoe

    def predict_target_bs_qoe(self, models, obs, bs_index, has_ho):
        cell_id = obs["cellId"][bs_index]
        ue_indexs = np.where(obs["servCellId"]==cell_id)[0]
        ho_ue_index = obs["hoUeIndex"]
        if has_ho:
            ue_indexs = ue_indexs[ue_indexs != ho_ue_index]
            ue_indexs = np.append(ue_indexs, ho_ue_index)
        bs_qoe = self.predict_bs_qoe(models, obs, bs_index, ue_indexs)
        return bs_qoe

    def new_action(self, models, obs, candidate_bs_indexs):
        t = obs["simTime"][0]
        epsilon = 2**(-t/60)
        print(t, epsilon)
        if self.rng.random() < epsilon:
            print("random choice")
            action = self.rng.choice(candidate_bs_indexs)
            return action

        serv_cell_index = obs["servCellIndex"]
        serv_bs_qoe_before = self.predict_serv_bs_qoe(models, obs, False)
        serv_bs_qoe_after = self.predict_serv_bs_qoe(models, obs, True)
        margins = []
        for bs_index in candidate_bs_indexs:
            if bs_index == serv_cell_index:
                margin = 0
                margins.append(margin)
                continue
            target_bs_qoe_before = self.predict_target_bs_qoe(models, obs, bs_index, False)
            target_bs_qoe_after = self.predict_target_bs_qoe(models, obs, bs_index, True)
            margin = (target_bs_qoe_after + serv_bs_qoe_after) - (target_bs_qoe_before + serv_bs_qoe_before)
            margins.append (margin)
        margin_array = np.array(margins)
        action = candidate_bs_indexs[np.argmax(margin_array)]
        return action

    def run(self):
        env = self.env
        nr_bs = self.args.nrBs
        prev_obss = {}

        models = self.new_models(nr_bs)
        buffers = self.new_buffers(nr_bs)
        data_iters = self.new_data_iters(buffers)
        #self.add_init_data_to_buffers(buffers, nr_bs)

        with self.open_file() as f:
            self.write_columns(f)

            obs, reward, done, info = env.get_state()
            print(obs, reward, done, info)
            while not done:
                ue_index = obs["hoUeIndex"]

                if ue_index in prev_obss:
                    prev_obs = prev_obss[ue_index]
                    self.write_prev_obs_reward(f, prev_obs, reward)
                    # add batch to buffer and train model
                    self.add_batch_to_buffers(buffers, prev_obs, reward)
                    self.train_model(models, buffers, data_iters, prev_obs.cellIndex)

                candidate_bs_indexs = self.new_candidate_bs_indexs(obs)
                if len(candidate_bs_indexs) == 0:
                    action = obs["servCellIndex"]
                elif len(candidate_bs_indexs) == 1:
                    action = candidate_bs_indexs[0]
                else:
                    action = self.new_action(models, obs, candidate_bs_indexs)

                print("action:", action, "rsrq:", obs["rsrq"][ue_index][action])
                print(self.args.agentName)
                print(self.args.bsInfoFn)
                print(self.args.obsUeInfoFn)
                print(self.args.obsUeTraceFn)

                prev_obss[ue_index] = self.create_prev_obs(obs, action)

                obs, reward, done, info = env.step(action)
                print(obs, reward, done, info)

                self.time_step += 1
                if (self.time_step % self.save_period) == 0:
                    for i in range(nr_bs):
                        self.save_bs_model_weights(models, i)

        # save model weight
        for i in range(nr_bs):
            self.save_bs_model_weights(models, i)


class RlAllModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 5

        self.bitrate_demand_denom = 1000000
        self.rsrq_bias = 12
        self.rsrq_denom = 9
        self.new_data_ratio_denom = 1
        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        self.max_nr_demand_ues = np.array([13,14,12,13,13])
        self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            prev_obs.bitrateDemand / self.bitrate_demand_denom,
            (prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            prev_obs.newDataRatio / self.new_data_ratio_denom,
            nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / self.bitrate_demand_denom
        f2 = (df2[["rsrq"]].to_numpy() + self.rsrq_bias) / self.rsrq_denom
        f3 = df2[["newDataRatio"]].to_numpy() / self.new_data_ratio_denom
        f4 = sum([df2[[f"nrUeForDemandType{i}"]].to_numpy()*self.demand_nr_txs[i] for i in range(0,5)]) / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
        f5 = df2[["mcs"]].to_numpy() / self.mcs_denom
        features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlOnlyDemandModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 1

        self.bitrate_demand_denom = 1000000
        #self.rsrq_bias = 12
        #self.rsrq_denom = 9
        #self.new_data_ratio_denom = 1
        #self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        #self.max_nr_demand_ues = np.array([13,14,12,13,13])
        #self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            #rsrq = obs["rsrq"][ue_index, bs_index]
            #new_data_ratio = obs["newDataRatio"][bs_index]
            #demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            #nr_ues = np.zeros(5)
            #for some_ue_index in ue_indexs:
            #    nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            #if ue_index != ho_ue_index:
            #    mcs = obs["mcs"][ue_index]
            #elif bs_index == obs["servCellIndex"]:
            #    mcs = obs["mcs"][ue_index]
            #else:
            #    other_indexs = ue_indexs[ue_indexs != ho_ue_index]
            #    rsrps = obs["rsrp"][other_indexs, bs_index]
            #    if len(rsrps) == 0:
            #        mcs = 30
            #    else:
            #        rsrp = obs["rsrp"][ho_ue_index, bs_index]
            #        closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
            #        mcss = obs["mcs"][other_indexs]
            #        mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            #rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            #new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            #nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            #mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                bitrate_demand, 
                #rsrq, 
                #new_data_ratio, 
                #nr_tx,
                #mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        #nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            prev_obs.bitrateDemand / self.bitrate_demand_denom,
            #(prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            #prev_obs.newDataRatio / self.new_data_ratio_denom,
            #nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            #prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlOnlyRsrqModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 1

        #self.bitrate_demand_denom = 1000000
        self.rsrq_bias = 12
        self.rsrq_denom = 9
        #self.new_data_ratio_denom = 1
        #self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        #self.max_nr_demand_ues = np.array([13,14,12,13,13])
        #self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            #bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            #new_data_ratio = obs["newDataRatio"][bs_index]
            #demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            #nr_ues = np.zeros(5)
            #for some_ue_index in ue_indexs:
            #    nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            #if ue_index != ho_ue_index:
            #    mcs = obs["mcs"][ue_index]
            #elif bs_index == obs["servCellIndex"]:
            #    mcs = obs["mcs"][ue_index]
            #else:
            #    other_indexs = ue_indexs[ue_indexs != ho_ue_index]
            #    rsrps = obs["rsrp"][other_indexs, bs_index]
            #    if len(rsrps) == 0:
            #        mcs = 30
            #    else:
            #        rsrp = obs["rsrp"][ho_ue_index, bs_index]
            #        closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
            #        mcss = obs["mcs"][other_indexs]
            #        mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            #bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            #new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            #nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            #mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                #bitrate_demand, 
                rsrq, 
                #new_data_ratio, 
                #nr_tx,
                #mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        #nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            #prev_obs.bitrateDemand / self.bitrate_demand_denom,
            (prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            #prev_obs.newDataRatio / self.new_data_ratio_denom,
            #nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            #prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlOnlyNewRatioModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 1

        #self.bitrate_demand_denom = 1000000
        #self.rsrq_bias = 12
        #self.rsrq_denom = 9
        self.new_data_ratio_denom = 1
        #self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        #self.max_nr_demand_ues = np.array([13,14,12,13,13])
        #self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            #bitrate_demand = obs["bitrateDemand"][ue_index]
            #rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            #demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            #nr_ues = np.zeros(5)
            #for some_ue_index in ue_indexs:
            #    nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            #if ue_index != ho_ue_index:
            #    mcs = obs["mcs"][ue_index]
            #elif bs_index == obs["servCellIndex"]:
            #    mcs = obs["mcs"][ue_index]
            #else:
            #    other_indexs = ue_indexs[ue_indexs != ho_ue_index]
            #    rsrps = obs["rsrp"][other_indexs, bs_index]
            #    if len(rsrps) == 0:
            #        mcs = 30
            #    else:
            #        rsrp = obs["rsrp"][ho_ue_index, bs_index]
            #        closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
            #        mcss = obs["mcs"][other_indexs]
            #        mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            #bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            #rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            #nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            #mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                #bitrate_demand, 
                #rsrq, 
                new_data_ratio, 
                #nr_tx,
                #mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        #nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            #prev_obs.bitrateDemand / self.bitrate_demand_denom,
            #(prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            prev_obs.newDataRatio / self.new_data_ratio_denom,
            #nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            #prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlOnlyNrUeModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 1

        #self.bitrate_demand_denom = 1000000
        #self.rsrq_bias = 12
        #self.rsrq_denom = 9
        #self.new_data_ratio_denom = 1
        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        self.max_nr_demand_ues = np.array([13,14,12,13,13])
        #self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            #bitrate_demand = obs["bitrateDemand"][ue_index]
            #rsrq = obs["rsrq"][ue_index, bs_index]
            #new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            #if ue_index != ho_ue_index:
            #    mcs = obs["mcs"][ue_index]
            #elif bs_index == obs["servCellIndex"]:
            #    mcs = obs["mcs"][ue_index]
            #else:
            #    other_indexs = ue_indexs[ue_indexs != ho_ue_index]
            #    rsrps = obs["rsrp"][other_indexs, bs_index]
            #    if len(rsrps) == 0:
            #        mcs = 30
            #    else:
            #        rsrp = obs["rsrp"][ho_ue_index, bs_index]
            #        closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
            #        mcss = obs["mcs"][other_indexs]
            #        mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            #bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            #rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            #new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            #mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                #bitrate_demand, 
                #rsrq, 
                #new_data_ratio, 
                nr_tx,
                #mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            #prev_obs.bitrateDemand / self.bitrate_demand_denom,
            #(prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            #prev_obs.newDataRatio / self.new_data_ratio_denom,
            nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            #prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlOnlyMcsModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 1

        #self.bitrate_demand_denom = 1000000
        #self.rsrq_bias = 12
        #self.rsrq_denom = 9
        #self.new_data_ratio_denom = 1
        #self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        #self.max_nr_demand_ues = np.array([13,14,12,13,13])
        self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            #bitrate_demand = obs["bitrateDemand"][ue_index]
            #rsrq = obs["rsrq"][ue_index, bs_index]
            #new_data_ratio = obs["newDataRatio"][bs_index]
            #demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            #nr_ues = np.zeros(5)
            #for some_ue_index in ue_indexs:
            #    nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            #bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            #rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            #new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            #nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                #bitrate_demand, 
                #rsrq, 
                #new_data_ratio, 
                #nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        #nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            #prev_obs.bitrateDemand / self.bitrate_demand_denom,
            #(prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            #prev_obs.newDataRatio / self.new_data_ratio_denom,
            #nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)


class RlNoDemandModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 4

        #self.bitrate_demand_denom = 1000000
        self.rsrq_bias = 12
        self.rsrq_denom = 9
        self.new_data_ratio_denom = 1
        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        self.max_nr_demand_ues = np.array([13,14,12,13,13])
        self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            #bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            #bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                #bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            #prev_obs.bitrateDemand / self.bitrate_demand_denom,
            (prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            prev_obs.newDataRatio / self.new_data_ratio_denom,
            nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        #f1 = df2[["bitrateDemand"]].to_numpy() / self.bitrate_demand_denom
        f2 = (df2[["rsrq"]].to_numpy() + self.rsrq_bias) / self.rsrq_denom
        f3 = df2[["newDataRatio"]].to_numpy() / self.new_data_ratio_denom
        f4 = sum([df2[[f"nrUeForDemandType{i}"]].to_numpy()*self.demand_nr_txs[i] for i in range(0,5)]) / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
        f5 = df2[["mcs"]].to_numpy() / self.mcs_denom
        #features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        features = np.concatenate([f2,f3,f4,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlNoRsrqModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 4

        self.bitrate_demand_denom = 1000000
        #self.rsrq_bias = 12
        #self.rsrq_denom = 9
        self.new_data_ratio_denom = 1
        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        self.max_nr_demand_ues = np.array([13,14,12,13,13])
        self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            #rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            #rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                bitrate_demand, 
                #rsrq, 
                new_data_ratio, 
                nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            prev_obs.bitrateDemand / self.bitrate_demand_denom,
            #(prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            prev_obs.newDataRatio / self.new_data_ratio_denom,
            nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / self.bitrate_demand_denom
        #f2 = (df2[["rsrq"]].to_numpy() + self.rsrq_bias) / self.rsrq_denom
        f3 = df2[["newDataRatio"]].to_numpy() / self.new_data_ratio_denom
        f4 = sum([df2[[f"nrUeForDemandType{i}"]].to_numpy()*self.demand_nr_txs[i] for i in range(0,5)]) / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
        f5 = df2[["mcs"]].to_numpy() / self.mcs_denom
        #features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        features = np.concatenate([f1,f3,f4,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlNoNewRatioModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 4

        self.bitrate_demand_denom = 1000000
        self.rsrq_bias = 12
        self.rsrq_denom = 9
        #self.new_data_ratio_denom = 1
        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        self.max_nr_demand_ues = np.array([13,14,12,13,13])
        self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            #new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            #new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                #new_data_ratio, 
                nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            prev_obs.bitrateDemand / self.bitrate_demand_denom,
            (prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            #prev_obs.newDataRatio / self.new_data_ratio_denom,
            nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / self.bitrate_demand_denom
        f2 = (df2[["rsrq"]].to_numpy() + self.rsrq_bias) / self.rsrq_denom
        #f3 = df2[["newDataRatio"]].to_numpy() / self.new_data_ratio_denom
        f4 = sum([df2[[f"nrUeForDemandType{i}"]].to_numpy()*self.demand_nr_txs[i] for i in range(0,5)]) / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
        f5 = df2[["mcs"]].to_numpy() / self.mcs_denom
        #features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        features = np.concatenate([f1,f2,f4,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlNoNrUeModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 4

        self.bitrate_demand_denom = 1000000
        self.rsrq_bias = 12
        self.rsrq_denom = 9
        self.new_data_ratio_denom = 1
        #self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        #self.max_nr_demand_ues = np.array([13,14,12,13,13])
        self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            #demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            #nr_ues = np.zeros(5)
            #for some_ue_index in ue_indexs:
            #    nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            #nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                #nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        #nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            prev_obs.bitrateDemand / self.bitrate_demand_denom,
            (prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            prev_obs.newDataRatio / self.new_data_ratio_denom,
            #nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / self.bitrate_demand_denom
        f2 = (df2[["rsrq"]].to_numpy() + self.rsrq_bias) / self.rsrq_denom
        f3 = df2[["newDataRatio"]].to_numpy() / self.new_data_ratio_denom
        #f4 = sum([df2[[f"nrUeForDemandType{i}"]].to_numpy()*self.demand_nr_txs[i] for i in range(0,5)]) / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
        f5 = df2[["mcs"]].to_numpy() / self.mcs_denom
        #features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        features = np.concatenate([f1,f2,f3,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlNoMcsModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 4

        self.bitrate_demand_denom = 1000000
        self.rsrq_bias = 12
        self.rsrq_denom = 9
        self.new_data_ratio_denom = 1
        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        self.max_nr_demand_ues = np.array([13,14,12,13,13])
        #self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            #if ue_index != ho_ue_index:
            #    mcs = obs["mcs"][ue_index]
            #elif bs_index == obs["servCellIndex"]:
            #    mcs = obs["mcs"][ue_index]
            #else:
            #    other_indexs = ue_indexs[ue_indexs != ho_ue_index]
            #    rsrps = obs["rsrp"][other_indexs, bs_index]
            #    if len(rsrps) == 0:
            #        mcs = 30
            #    else:
            #        rsrp = obs["rsrp"][ho_ue_index, bs_index]
            #        closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
            #        mcss = obs["mcs"][other_indexs]
            #        mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            #mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                nr_tx,
                #mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            prev_obs.bitrateDemand / self.bitrate_demand_denom,
            (prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            prev_obs.newDataRatio / self.new_data_ratio_denom,
            nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            #prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / self.bitrate_demand_denom
        f2 = (df2[["rsrq"]].to_numpy() + self.rsrq_bias) / self.rsrq_denom
        f3 = df2[["newDataRatio"]].to_numpy() / self.new_data_ratio_denom
        f4 = sum([df2[[f"nrUeForDemandType{i}"]].to_numpy()*self.demand_nr_txs[i] for i in range(0,5)]) / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
        #f5 = df2[["mcs"]].to_numpy() / self.mcs_denom
        #features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        features = np.concatenate([f1,f2,f3,f4], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlNoNewRatioNrUeModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 3

        self.bitrate_demand_denom = 1000000
        self.rsrq_bias = 12
        self.rsrq_denom = 9
        #self.new_data_ratio_denom = 1
        #self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        #self.max_nr_demand_ues = np.array([13,14,12,13,13])
        self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            #new_data_ratio = obs["newDataRatio"][bs_index]
            #demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            #nr_ues = np.zeros(5)
            #for some_ue_index in ue_indexs:
            #    nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            #new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            #nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                #new_data_ratio, 
                #nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        #nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            prev_obs.bitrateDemand / self.bitrate_demand_denom,
            (prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            #prev_obs.newDataRatio / self.new_data_ratio_denom,
            #nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / self.bitrate_demand_denom
        f2 = (df2[["rsrq"]].to_numpy() + self.rsrq_bias) / self.rsrq_denom
        #f3 = df2[["newDataRatio"]].to_numpy() / self.new_data_ratio_denom
        #f4 = sum([df2[[f"nrUeForDemandType{i}"]].to_numpy()*self.demand_nr_txs[i] for i in range(0,5)]) / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
        f5 = df2[["mcs"]].to_numpy() / self.mcs_denom
        #features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        features = np.concatenate([f1,f2,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlNoRsrqMcsModelG12SaveModelT8(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 3

        self.bitrate_demand_denom = 1000000
        #self.rsrq_bias = 12
        #self.rsrq_denom = 9
        self.new_data_ratio_denom = 1
        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        self.max_nr_demand_ues = np.array([13,14,12,13,13])
        #self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            #rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            #if ue_index != ho_ue_index:
            #    mcs = obs["mcs"][ue_index]
            #elif bs_index == obs["servCellIndex"]:
            #    mcs = obs["mcs"][ue_index]
            #else:
            #    other_indexs = ue_indexs[ue_indexs != ho_ue_index]
            #    rsrps = obs["rsrp"][other_indexs, bs_index]
            #    if len(rsrps) == 0:
            #        mcs = 30
            #    else:
            #        rsrp = obs["rsrp"][ho_ue_index, bs_index]
            #        closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
            #        mcss = obs["mcs"][other_indexs]
            #        mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            #rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            #mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                bitrate_demand, 
                #rsrq, 
                new_data_ratio, 
                nr_tx,
                #mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            prev_obs.bitrateDemand / self.bitrate_demand_denom,
            #(prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            prev_obs.newDataRatio / self.new_data_ratio_denom,
            nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            #prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlAllModelG12SaveModelT11(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 5

        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])

        self.bitrate_demand_tics = [100000, 160000, 300000, 500000, 1000000]
        self.rsrq_tics = [-12, -9, -8, -7, -3]
        self.new_data_ratio_tics = [0, 0.98, 1]
        self.nr_tx_tics = [12.5, 200, 785, 1145, 2905]
        self.mcs_tics = [0, 5, 8, 27, 30]

    def scale_feature(self, tics, feature):
        d = 1 / (len(tics)-1)
        for i, tic in enumerate(tics):
            if feature < tic:
                break
        else:
            return 1.0
        scaled_feature = d*(i-1) + d*(feature-tics[i-1])/(tics[i]-tics[i-1])
        return scaled_feature

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = self.scale_feature(self.bitrate_demand_tics, bitrate_demand)
            rsrq = self.scale_feature(self.rsrq_tics, rsrq)
            new_data_ratio = self.scale_feature(self.new_data_ratio_tics, new_data_ratio)
            nr_tx = self.scale_feature(self.nr_tx_tics, nr_tx)
            mcs = self.scale_feature(self.mcs_tics, mcs)
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            self.scale_feature(self.bitrate_demand_tics, prev_obs.bitrateDemand),
            self.scale_feature(self.rsrq_tics, prev_obs.rsrq),
            self.scale_feature(self.new_data_ratio_tics, prev_obs.newDataRatio),
            self.scale_feature(self.nr_tx_tics, nr_tx),
            self.scale_feature(self.mcs_tics, prev_obs.mcs),
        ])

        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlNoDemandModelG12SaveModelT11(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 4

        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])

        #self.bitrate_demand_tics = [100000, 160000, 300000, 500000, 1000000]
        self.rsrq_tics = [-12, -9, -8, -7, -3]
        self.new_data_ratio_tics = [0, 0.98, 1]
        self.nr_tx_tics = [12.5, 200, 785, 1145, 2905]
        self.mcs_tics = [0, 5, 8, 27, 30]

    def scale_feature(self, tics, feature):
        d = 1 / (len(tics)-1)
        for i, tic in enumerate(tics):
            if feature < tic:
                break
        else:
            return 1.0
        scaled_feature = d*(i-1) + d*(feature-tics[i-1])/(tics[i]-tics[i-1])
        return scaled_feature

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            #bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            #bitrate_demand = self.scale_feature(self.bitrate_demand_tics, bitrate_demand)
            rsrq = self.scale_feature(self.rsrq_tics, rsrq)
            new_data_ratio = self.scale_feature(self.new_data_ratio_tics, new_data_ratio)
            nr_tx = self.scale_feature(self.nr_tx_tics, nr_tx)
            mcs = self.scale_feature(self.mcs_tics, mcs)
            feature_vec = np.array([
                #bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            #self.scale_feature(self.bitrate_demand_tics, prev_obs.bitrateDemand),
            self.scale_feature(self.rsrq_tics, prev_obs.rsrq),
            self.scale_feature(self.new_data_ratio_tics, prev_obs.newDataRatio),
            self.scale_feature(self.nr_tx_tics, nr_tx),
            self.scale_feature(self.mcs_tics, prev_obs.mcs),
        ])

        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlNoRsrqModelG12SaveModelT11(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 4

        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])

        self.bitrate_demand_tics = [100000, 160000, 300000, 500000, 1000000]
        #self.rsrq_tics = [-12, -9, -8, -7, -3]
        self.new_data_ratio_tics = [0, 0.98, 1]
        self.nr_tx_tics = [12.5, 200, 785, 1145, 2905]
        self.mcs_tics = [0, 5, 8, 27, 30]

    def scale_feature(self, tics, feature):
        d = 1 / (len(tics)-1)
        for i, tic in enumerate(tics):
            if feature < tic:
                break
        else:
            return 1.0
        scaled_feature = d*(i-1) + d*(feature-tics[i-1])/(tics[i]-tics[i-1])
        return scaled_feature

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            #rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = self.scale_feature(self.bitrate_demand_tics, bitrate_demand)
            #rsrq = self.scale_feature(self.rsrq_tics, rsrq)
            new_data_ratio = self.scale_feature(self.new_data_ratio_tics, new_data_ratio)
            nr_tx = self.scale_feature(self.nr_tx_tics, nr_tx)
            mcs = self.scale_feature(self.mcs_tics, mcs)
            feature_vec = np.array([
                bitrate_demand, 
                #rsrq, 
                new_data_ratio, 
                nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            self.scale_feature(self.bitrate_demand_tics, prev_obs.bitrateDemand),
            #self.scale_feature(self.rsrq_tics, prev_obs.rsrq),
            self.scale_feature(self.new_data_ratio_tics, prev_obs.newDataRatio),
            self.scale_feature(self.nr_tx_tics, nr_tx),
            self.scale_feature(self.mcs_tics, prev_obs.mcs),
        ])

        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlNoNewRatioModelG12SaveModelT11(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 4

        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])

        self.bitrate_demand_tics = [100000, 160000, 300000, 500000, 1000000]
        self.rsrq_tics = [-12, -9, -8, -7, -3]
        #self.new_data_ratio_tics = [0, 0.98, 1]
        self.nr_tx_tics = [12.5, 200, 785, 1145, 2905]
        self.mcs_tics = [0, 5, 8, 27, 30]

    def scale_feature(self, tics, feature):
        d = 1 / (len(tics)-1)
        for i, tic in enumerate(tics):
            if feature < tic:
                break
        else:
            return 1.0
        scaled_feature = d*(i-1) + d*(feature-tics[i-1])/(tics[i]-tics[i-1])
        return scaled_feature

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            #new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = self.scale_feature(self.bitrate_demand_tics, bitrate_demand)
            rsrq = self.scale_feature(self.rsrq_tics, rsrq)
            #new_data_ratio = self.scale_feature(self.new_data_ratio_tics, new_data_ratio)
            nr_tx = self.scale_feature(self.nr_tx_tics, nr_tx)
            mcs = self.scale_feature(self.mcs_tics, mcs)
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                #new_data_ratio, 
                nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            self.scale_feature(self.bitrate_demand_tics, prev_obs.bitrateDemand),
            self.scale_feature(self.rsrq_tics, prev_obs.rsrq),
            #self.scale_feature(self.new_data_ratio_tics, prev_obs.newDataRatio),
            self.scale_feature(self.nr_tx_tics, nr_tx),
            self.scale_feature(self.mcs_tics, prev_obs.mcs),
        ])

        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlNoNrUeModelG12SaveModelT11(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 4

        #self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])

        self.bitrate_demand_tics = [100000, 160000, 300000, 500000, 1000000]
        self.rsrq_tics = [-12, -9, -8, -7, -3]
        self.new_data_ratio_tics = [0, 0.98, 1]
        #self.nr_tx_tics = [12.5, 200, 785, 1145, 2905]
        self.mcs_tics = [0, 5, 8, 27, 30]

    def scale_feature(self, tics, feature):
        d = 1 / (len(tics)-1)
        for i, tic in enumerate(tics):
            if feature < tic:
                break
        else:
            return 1.0
        scaled_feature = d*(i-1) + d*(feature-tics[i-1])/(tics[i]-tics[i-1])
        return scaled_feature

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            #demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            #nr_ues = np.zeros(5)
            #for some_ue_index in ue_indexs:
            #    nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = self.scale_feature(self.bitrate_demand_tics, bitrate_demand)
            rsrq = self.scale_feature(self.rsrq_tics, rsrq)
            new_data_ratio = self.scale_feature(self.new_data_ratio_tics, new_data_ratio)
            #nr_tx = self.scale_feature(self.nr_tx_tics, nr_tx)
            mcs = self.scale_feature(self.mcs_tics, mcs)
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                #nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        #nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            self.scale_feature(self.bitrate_demand_tics, prev_obs.bitrateDemand),
            self.scale_feature(self.rsrq_tics, prev_obs.rsrq),
            self.scale_feature(self.new_data_ratio_tics, prev_obs.newDataRatio),
            #self.scale_feature(self.nr_tx_tics, nr_tx),
            self.scale_feature(self.mcs_tics, prev_obs.mcs),
        ])

        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlNoMcsModelG12SaveModelT11(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 4

        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])

        self.bitrate_demand_tics = [100000, 160000, 300000, 500000, 1000000]
        self.rsrq_tics = [-12, -9, -8, -7, -3]
        self.new_data_ratio_tics = [0, 0.98, 1]
        self.nr_tx_tics = [12.5, 200, 785, 1145, 2905]
        #self.mcs_tics = [0, 5, 8, 27, 30]

    def scale_feature(self, tics, feature):
        d = 1 / (len(tics)-1)
        for i, tic in enumerate(tics):
            if feature < tic:
                break
        else:
            return 1.0
        scaled_feature = d*(i-1) + d*(feature-tics[i-1])/(tics[i]-tics[i-1])
        return scaled_feature

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            #if ue_index != ho_ue_index:
            #    mcs = obs["mcs"][ue_index]
            #elif bs_index == obs["servCellIndex"]:
            #    mcs = obs["mcs"][ue_index]
            #else:
            #    other_indexs = ue_indexs[ue_indexs != ho_ue_index]
            #    rsrps = obs["rsrp"][other_indexs, bs_index]
            #    if len(rsrps) == 0:
            #        mcs = 30
            #    else:
            #        rsrp = obs["rsrp"][ho_ue_index, bs_index]
            #        closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
            #        mcss = obs["mcs"][other_indexs]
            #        mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = self.scale_feature(self.bitrate_demand_tics, bitrate_demand)
            rsrq = self.scale_feature(self.rsrq_tics, rsrq)
            new_data_ratio = self.scale_feature(self.new_data_ratio_tics, new_data_ratio)
            nr_tx = self.scale_feature(self.nr_tx_tics, nr_tx)
            #mcs = self.scale_feature(self.mcs_tics, mcs)
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                nr_tx,
                #mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            self.scale_feature(self.bitrate_demand_tics, prev_obs.bitrateDemand),
            self.scale_feature(self.rsrq_tics, prev_obs.rsrq),
            self.scale_feature(self.new_data_ratio_tics, prev_obs.newDataRatio),
            self.scale_feature(self.nr_tx_tics, nr_tx),
            #self.scale_feature(self.mcs_tics, prev_obs.mcs),
        ])

        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlNoNewRatioNrUeModelG12SaveModelT11(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 3

        #self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])

        self.bitrate_demand_tics = [100000, 160000, 300000, 500000, 1000000]
        self.rsrq_tics = [-12, -9, -8, -7, -3]
        #self.new_data_ratio_tics = [0, 0.98, 1]
        #self.nr_tx_tics = [12.5, 200, 785, 1145, 2905]
        self.mcs_tics = [0, 5, 8, 27, 30]

    def scale_feature(self, tics, feature):
        d = 1 / (len(tics)-1)
        for i, tic in enumerate(tics):
            if feature < tic:
                break
        else:
            return 1.0
        scaled_feature = d*(i-1) + d*(feature-tics[i-1])/(tics[i]-tics[i-1])
        return scaled_feature

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            #new_data_ratio = obs["newDataRatio"][bs_index]
            #demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            #nr_ues = np.zeros(5)
            #for some_ue_index in ue_indexs:
            #    nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = self.scale_feature(self.bitrate_demand_tics, bitrate_demand)
            rsrq = self.scale_feature(self.rsrq_tics, rsrq)
            #new_data_ratio = self.scale_feature(self.new_data_ratio_tics, new_data_ratio)
            #nr_tx = self.scale_feature(self.nr_tx_tics, nr_tx)
            mcs = self.scale_feature(self.mcs_tics, mcs)
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                #new_data_ratio, 
                #nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        #nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        #nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            self.scale_feature(self.bitrate_demand_tics, prev_obs.bitrateDemand),
            self.scale_feature(self.rsrq_tics, prev_obs.rsrq),
            #self.scale_feature(self.new_data_ratio_tics, prev_obs.newDataRatio),
            #self.scale_feature(self.nr_tx_tics, nr_tx),
            self.scale_feature(self.mcs_tics, prev_obs.mcs),
        ])

        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class RlNoRsrqMcsModelG12SaveModelT11(RlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 3

        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])

        self.bitrate_demand_tics = [100000, 160000, 300000, 500000, 1000000]
        #self.rsrq_tics = [-12, -9, -8, -7, -3]
        self.new_data_ratio_tics = [0, 0.98, 1]
        self.nr_tx_tics = [12.5, 200, 785, 1145, 2905]
        #self.mcs_tics = [0, 5, 8, 27, 30]

    def scale_feature(self, tics, feature):
        d = 1 / (len(tics)-1)
        for i, tic in enumerate(tics):
            if feature < tic:
                break
        else:
            return 1.0
        scaled_feature = d*(i-1) + d*(feature-tics[i-1])/(tics[i]-tics[i-1])
        return scaled_feature

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            #rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            #if ue_index != ho_ue_index:
            #    mcs = obs["mcs"][ue_index]
            #elif bs_index == obs["servCellIndex"]:
            #    mcs = obs["mcs"][ue_index]
            #else:
            #    other_indexs = ue_indexs[ue_indexs != ho_ue_index]
            #    rsrps = obs["rsrp"][other_indexs, bs_index]
            #    if len(rsrps) == 0:
            #        mcs = 30
            #    else:
            #        rsrp = obs["rsrp"][ho_ue_index, bs_index]
            #        closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
            #        mcss = obs["mcs"][other_indexs]
            #        mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = self.scale_feature(self.bitrate_demand_tics, bitrate_demand)
            #rsrq = self.scale_feature(self.rsrq_tics, rsrq)
            new_data_ratio = self.scale_feature(self.new_data_ratio_tics, new_data_ratio)
            nr_tx = self.scale_feature(self.nr_tx_tics, nr_tx)
            #mcs = self.scale_feature(self.mcs_tics, mcs)
            feature_vec = np.array([
                bitrate_demand, 
                #rsrq, 
                new_data_ratio, 
                nr_tx,
                #mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            self.scale_feature(self.bitrate_demand_tics, prev_obs.bitrateDemand),
            #self.scale_feature(self.rsrq_tics, prev_obs.rsrq),
            self.scale_feature(self.new_data_ratio_tics, prev_obs.newDataRatio),
            self.scale_feature(self.nr_tx_tics, nr_tx),
            #self.scale_feature(self.mcs_tics, prev_obs.mcs),
        ])

        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

class QoeEventTriggerRlNnMaxQoeMarginBaseG12SaveModel(RlNnMaxQoeMarginBaseG12SaveModel):
    def run(self):
        env = self.env
        nr_bs = self.args.nrBs
        prev_obss = {}

        models = self.new_models(nr_bs)
        buffers = self.new_buffers(nr_bs)
        data_iters = self.new_data_iters(buffers)
        #self.add_init_data_to_buffers(buffers, nr_bs)

        with self.open_file() as f:
            self.write_columns(f)

            obs, reward, done, info = env.get_state()
            print(obs, reward, done, info)
            while not done:
                ue_index = obs["hoUeIndex"]

                if ue_index in prev_obss:
                    prev_obs = prev_obss[ue_index]
                    self.write_prev_obs_reward(f, prev_obs, reward)
                    # add batch to buffer and train model
                    self.add_batch_to_buffers(buffers, prev_obs, reward)
                    self.train_model(models, buffers, data_iters, prev_obs.cellIndex)

                candidate_bs_indexs = self.new_candidate_bs_indexs(obs)
                if len(candidate_bs_indexs) == 0:
                    action = obs["servCellIndex"]
                elif obs["qoe"][ue_index] >= self.qoe_threshold:
                    print("event is not triggered", obs["qoe"][ue_index], self.qoe_threshold)
                    action = obs["servCellIndex"]
                elif len(candidate_bs_indexs) == 1:
                    action = candidate_bs_indexs[0]
                else:
                    action = self.new_action(models, obs, candidate_bs_indexs)

                print("action:", action, "rsrq:", obs["rsrq"][ue_index][action])
                print(self.args.agentName)
                print(self.args.bsInfoFn)
                print(self.args.obsUeInfoFn)
                print(self.args.obsUeTraceFn)

                prev_obss[ue_index] = self.create_prev_obs(obs, action)

                obs, reward, done, info = env.step(action)
                print(obs, reward, done, info)

                self.time_step += 1
                if (self.time_step % self.save_period) == 0:
                    for i in range(nr_bs):
                        self.save_bs_model_weights(models, i)

        # save model weight
        for i in range(nr_bs):
            self.save_bs_model_weights(models, i)

class QoeEventTriggerRlAllModelG12SaveModelT8(QoeEventTriggerRlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 5

        self.bitrate_demand_denom = 1000000
        self.rsrq_bias = 12
        self.rsrq_denom = 9
        self.new_data_ratio_denom = 1
        self.demand_nr_txs = np.array([100/8, 160/8, 300/8, 500/8, 1000/8])
        self.max_nr_demand_ues = np.array([13,14,12,13,13])
        self.mcs_denom = 28

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            nr_tx = np.sum(self.demand_nr_txs * nr_ues)
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / self.bitrate_demand_denom
            rsrq = (rsrq + self.rsrq_bias) / self.rsrq_denom
            new_data_ratio = new_data_ratio / self.new_data_ratio_denom
            nr_tx = nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
            mcs = mcs / self.mcs_denom
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                nr_tx,
                mcs,
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        nr_ues = np.array([getattr(prev_obs, f"nrUeForDemandType{i}") for i in range(5)])
        nr_tx = np.sum(self.demand_nr_txs * nr_ues)
        feature_array = np.array([
            prev_obs.bitrateDemand / self.bitrate_demand_denom,
            (prev_obs.rsrq + self.rsrq_bias) / self.rsrq_denom,
            prev_obs.newDataRatio / self.new_data_ratio_denom,
            nr_tx / np.sum(self.demand_nr_txs * self.max_nr_demand_ues),
            prev_obs.mcs / self.mcs_denom,
        ])
        print("feature_array:", feature_array)
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / self.bitrate_demand_denom
        f2 = (df2[["rsrq"]].to_numpy() + self.rsrq_bias) / self.rsrq_denom
        f3 = df2[["newDataRatio"]].to_numpy() / self.new_data_ratio_denom
        f4 = sum([df2[[f"nrUeForDemandType{i}"]].to_numpy()*self.demand_nr_txs[i] for i in range(0,5)]) / np.sum(self.demand_nr_txs * self.max_nr_demand_ues)
        f5 = df2[["mcs"]].to_numpy() / self.mcs_denom
        features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels

class RlSimpleFeatureModelG12(QoeEventTriggerRlNnMaxQoeMarginBaseG12SaveModel):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.input_shape = 9

    def predict_bs_qoe(self, models, obs, bs_index, ue_indexs):
        if len(ue_indexs) == 0:
            return 0
        model = models[bs_index]
        ho_ue_index = obs["hoUeIndex"]
        feature_vecs = []
        for ue_index in ue_indexs:
            bitrate_demand = obs["bitrateDemand"][ue_index]
            rsrq = obs["rsrq"][ue_index, bs_index]
            new_data_ratio = obs["newDataRatio"][bs_index]
            demand_indexs = {100000: 0, 160000: 1, 300000: 2, 500000: 3, 1000000: 4}
            nr_ues = np.zeros(5)
            for some_ue_index in ue_indexs:
                nr_ues[demand_indexs[obs["bitrateDemand"][some_ue_index]]] += 1
            if ue_index != ho_ue_index:
                mcs = obs["mcs"][ue_index]
            elif bs_index == obs["servCellIndex"]:
                mcs = obs["mcs"][ue_index]
            else:
                other_indexs = ue_indexs[ue_indexs != ho_ue_index]
                rsrps = obs["rsrp"][other_indexs, bs_index]
                if len(rsrps) == 0:
                    mcs = 30
                else:
                    rsrp = obs["rsrp"][ho_ue_index, bs_index]
                    closest_mcs_indexs = np.argsort(np.abs(rsrps - rsrp))
                    mcss = obs["mcs"][other_indexs]
                    mcs = np.mean(mcss[closest_mcs_indexs[0:3]])
            bitrate_demand = bitrate_demand / 1000000
            rsrq = rsrq / 12
            new_data_ratio = new_data_ratio / 1
            nr_ues = nr_ues / 10
            mcs = mcs / 28
            feature_vec = np.array([
                bitrate_demand, 
                rsrq, 
                new_data_ratio, 
                *nr_ues, 
                mcs
            ])
            feature_vecs.append(feature_vec)
        feature_array = np.stack(feature_vecs)
        predict_qoes = model.predict(feature_array)
        bs_qoe = np.sum(predict_qoes)
        return bs_qoe

    def new_data_spec(self, bs_index):
        env = self.env
        feature_shape = (self.input_shape, )
        feature_dtype = env.observation_space["rsrq"].dtype
        label_shape = (1, )
        label_dtype = feature_dtype
        data_spec = (specs.ArraySpec(feature_shape, feature_dtype, "feature"), 
                     specs.ArraySpec(label_shape, label_dtype, "label"))
        return data_spec

    def add_batch_to_buffers(self, buffers, prev_obs, reward):
        bs_index = prev_obs.cellIndex
        feature_array = np.array([
            prev_obs.bitrateDemand / 1000000,
            prev_obs.rsrq / 12,
            prev_obs.newDataRatio / 1,
            *[getattr(prev_obs, f"nrUeForDemandType{i}") / 10 for i in range(5)],
            prev_obs.mcs / 28,
        ])
        reward_array = np.zeros(1) + reward
        item = (np.expand_dims(feature_array, 0), np.expand_dims(reward_array, 0))
        buffer = buffers[bs_index]
        buffer.add_batch(item)

    def get_bs_features_and_labels(self, training_df, bs_index):
        df1 = training_df
        df2 = df1[df1["cellIndex"]==bs_index]
        if len(df2.index) == 0:
            return np.array([]), np.array([])
        f1 = df2[["bitrateDemand"]].to_numpy() / 1000000 
        f2 = df2[["rsrq"]].to_numpy() / 12 
        f3 = df2[["newDataRatio"]].to_numpy() / 1
        f4 = df2[[f"nrUeForDemandType{i}" for i in range(0,5)]].to_numpy() / 10
        f5 = df2[["mcs"]].to_numpy() / 28
        features = np.concatenate([f1,f2,f3,f4,f5], axis=1)
        # features = np.concatenate([f1,f3,f4,f5], axis=1)
        print(features.shape)
        labels = df2["qoeScore"].to_numpy()
        return features, labels
    def run(self):
        env = self.env
        nr_bs = self.args.nrBs

        prev_obss = {}

        models = self.new_models(nr_bs)
        buffers = self.new_buffers(nr_bs)
        data_iters = self.new_data_iters(buffers)
        #self.add_init_data_to_buffers(buffers, nr_bs)
        with self.open_file() as f:
            self.write_columns(f)
            obs, reward, done, info = env.get_state()
            print(obs, reward, done, info)
            iterationNum = int(self.args.iterations)
            currIt = 0
            while True:
                print("Start iteration: ", currIt)
                obs = env.reset()
                print("---obs:", obs)
                print("------------------------press any key to continue---------------------------")
                input()
                done = False
                while not done:
                    ue_index = obs["hoUeIndex"]

                    if ue_index in prev_obss:
                        prev_obs = prev_obss[ue_index]
                        self.write_prev_obs_reward(f, prev_obs, reward)
                        # add batch to buffer and train model
                        self.add_batch_to_buffers(buffers, prev_obs, reward)
                        self.train_model(models, buffers, data_iters, prev_obs.cellIndex)

                    candidate_bs_indexs = self.new_candidate_bs_indexs(obs)
                    if len(candidate_bs_indexs) == 0:
                        action = obs["servCellIndex"]
                    elif obs["qoe"][ue_index] >= self.qoe_threshold:
                        print("event is not triggered", obs["qoe"][ue_index], self.qoe_threshold)
                        action = obs["servCellIndex"]
                    elif len(candidate_bs_indexs) == 1:
                        action = candidate_bs_indexs[0]
                    else:
                        action = self.new_action(models, obs, candidate_bs_indexs)

                    print("action:", action, "rsrq:", obs["rsrq"][ue_index][action])
                    print(self.args.agentName)
                    print(self.args.bsInfoFn)
                    print(self.args.obsUeInfoFn)
                    print(self.args.obsUeTraceFn)

                    prev_obss[ue_index] = self.create_prev_obs(obs, action)

                    obs, reward, done, info = env.step(action)
                    print(obs, reward, done, info)

                    self.time_step += 1
                    if (self.time_step % self.save_period) == 0:
                        for i in range(nr_bs):
                            self.save_bs_model_weights(models, i)
                currIt += 1
                if currIt == iterationNum:
                    break

        # save model weight
        for i in range(nr_bs):
            self.save_bs_model_weights(models, i)


###
class RlNnMaxQoeMarginG12Model12100(RlAllModelG12):
    pass

class RlNnMaxQoeMarginG12Model12200(RlNoDemandModelG12):
    pass

class RlNnMaxQoeMarginG12Model12300(RlNoRsrqModelG12):
    pass

class RlNnMaxQoeMarginG12Model12400(RlNoNewRatioModelG12):
    pass

class RlNnMaxQoeMarginG12Model12500(RlNoNrUeModelG12):
    pass

class RlNnMaxQoeMarginG12Model12600(RlNoMcsModelG12):
    pass

###
class RlNnMaxQoeMarginG12Model12107(RlAllModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model12207(RlNoDemandModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model12307(RlNoRsrqModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model12407(RlNoNewRatioModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model12507(RlNoNrUeModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model12607(RlNoMcsModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model12707(RlNoNewRatioNrUeModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model12807(RlNoRsrqMcsModelG12SaveModelT8):
    pass

###
class RlNnMaxQoeMarginG12Model13100(RlAllModelG12):
    pass

class RlNnMaxQoeMarginG12Model13200(RlNoDemandModelG12):
    pass

class RlNnMaxQoeMarginG12Model13300(RlNoRsrqModelG12):
    pass

class RlNnMaxQoeMarginG12Model13400(RlNoNewRatioModelG12):
    pass

class RlNnMaxQoeMarginG12Model13500(RlNoNrUeModelG12):
    pass

class RlNnMaxQoeMarginG12Model13600(RlNoMcsModelG12):
    pass

###
class RlNnMaxQoeMarginG12Model11100(RlAllModelG12):
    pass

class RlNnMaxQoeMarginG12Model11200(RlNoDemandModelG12):
    pass

class RlNnMaxQoeMarginG12Model11300(RlNoRsrqModelG12):
    pass

class RlNnMaxQoeMarginG12Model11400(RlNoNewRatioModelG12):
    pass

class RlNnMaxQoeMarginG12Model11500(RlNoNrUeModelG12):
    pass

class RlNnMaxQoeMarginG12Model11600(RlNoMcsModelG12):
    pass

###
class RlNnMaxQoeMarginG12Model22100(RlAllModelG12):
    pass

class RlNnMaxQoeMarginG12Model22200(RlNoDemandModelG12):
    pass

class RlNnMaxQoeMarginG12Model22300(RlNoRsrqModelG12):
    pass

class RlNnMaxQoeMarginG12Model22400(RlNoNewRatioModelG12):
    pass

class RlNnMaxQoeMarginG12Model22500(RlNoNrUeModelG12):
    pass

class RlNnMaxQoeMarginG12Model22600(RlNoMcsModelG12):
    pass

###
class RlNnMaxQoeMarginG12Model23100(RlAllModelG12):
    pass

class RlNnMaxQoeMarginG12Model23200(RlNoDemandModelG12):
    pass

class RlNnMaxQoeMarginG12Model23300(RlNoRsrqModelG12):
    pass

class RlNnMaxQoeMarginG12Model23400(RlNoNewRatioModelG12):
    pass

class RlNnMaxQoeMarginG12Model23500(RlNoNrUeModelG12):
    pass

class RlNnMaxQoeMarginG12Model23600(RlNoMcsModelG12):
    pass

###
class RlNnMaxQoeMarginG12Model24107(RlAllModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model24207(RlNoDemandModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model24307(RlNoRsrqModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model24407(RlNoNewRatioModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model24507(RlNoNrUeModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model24607(RlNoMcsModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model24707(RlNoNewRatioNrUeModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model24807(RlNoRsrqMcsModelG12SaveModelT8):
    pass

###
class RlNnMaxQoeMarginG12Model24110(RlAllModelG12SaveModelT11):
    pass

class RlNnMaxQoeMarginG12Model24210(RlNoDemandModelG12SaveModelT11):
    pass

class RlNnMaxQoeMarginG12Model24310(RlNoRsrqModelG12SaveModelT11):
    pass

class RlNnMaxQoeMarginG12Model24410(RlNoNewRatioModelG12SaveModelT11):
    pass

class RlNnMaxQoeMarginG12Model24510(RlNoNrUeModelG12SaveModelT11):
    pass

class RlNnMaxQoeMarginG12Model24610(RlNoMcsModelG12SaveModelT11):
    pass

class RlNnMaxQoeMarginG12Model24710(RlNoNewRatioNrUeModelG12SaveModelT11):
    pass

class RlNnMaxQoeMarginG12Model24810(RlNoRsrqMcsModelG12SaveModelT11):
    pass


###
class RlNnMaxQoeMarginG12Model32100(RlAllModelG12):
    pass

class RlNnMaxQoeMarginG12Model32200(RlNoDemandModelG12):
    pass

class RlNnMaxQoeMarginG12Model32300(RlNoRsrqModelG12):
    pass

class RlNnMaxQoeMarginG12Model32400(RlNoNewRatioModelG12):
    pass

class RlNnMaxQoeMarginG12Model32500(RlNoNrUeModelG12):
    pass

class RlNnMaxQoeMarginG12Model32600(RlNoMcsModelG12):
    pass

###
class RlNnMaxQoeMarginG12Model33100(RlAllModelG12):
    pass

class RlNnMaxQoeMarginG12Model33200(RlNoDemandModelG12):
    pass

class RlNnMaxQoeMarginG12Model33300(RlNoRsrqModelG12):
    pass

class RlNnMaxQoeMarginG12Model33400(RlNoNewRatioModelG12):
    pass

class RlNnMaxQoeMarginG12Model33500(RlNoNrUeModelG12):
    pass

class RlNnMaxQoeMarginG12Model33600(RlNoMcsModelG12):
    pass

###
class RlNnMaxQoeMarginG12Model34107(RlAllModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model34207(RlNoDemandModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model34307(RlNoRsrqModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model34407(RlNoNewRatioModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model34507(RlNoNrUeModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model34607(RlNoMcsModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model34707(RlNoNewRatioNrUeModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model34807(RlNoRsrqMcsModelG12SaveModelT8):
    pass

###
class RlNnMaxQoeMarginG12Model01010100(RlAllModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model01010200(RlOnlyDemandModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model01010300(RlOnlyRsrqModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model01010400(RlOnlyNewRatioModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model01010500(RlOnlyNrUeModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model01010600(RlOnlyMcsModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model01010700(QoeEventTriggerRlAllModelG12SaveModelT8):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.qoe_threshold = 0.8

###
class RlNnMaxQoeMarginG12Model02010100(RlAllModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model02010200(RlOnlyDemandModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model02010300(RlOnlyRsrqModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model02010400(RlOnlyNewRatioModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model02010500(RlOnlyNrUeModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model02010600(RlOnlyMcsModelG12SaveModelT8):
    pass

###
class RlNnMaxQoeMarginG12Model03010100(RlAllModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model03010200(RlOnlyDemandModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model03010300(RlOnlyRsrqModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model03010400(RlOnlyNewRatioModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model03010500(RlOnlyNrUeModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model03010600(RlOnlyMcsModelG12SaveModelT8):
    pass

###
class RlNnMaxQoeMarginG12Model04010100(RlAllModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model04010200(RlOnlyDemandModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model04010300(RlOnlyRsrqModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model04010400(RlOnlyNewRatioModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model04010500(RlOnlyNrUeModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model04010600(RlOnlyMcsModelG12SaveModelT8):
    pass

###
class RlNnMaxQoeMarginG12Model05010100(RlAllModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model05010700(RlNoMcsModelG12SaveModelT8):
    pass

class RlNnMaxQoeMarginG12Model10311803(RlSimpleFeatureModelG12):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.qoe_threshold = 0.8



def run_policy(env, args):
    agentName = args.agentName
    print(agentName)
    if agentName == "none":
        pass
    elif agentName == "rlNnMaxQoeMarginG12Model12100":
        agent = RlNnMaxQoeMarginG12Model12100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12200":
        agent = RlNnMaxQoeMarginG12Model12200(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12300":
        agent = RlNnMaxQoeMarginG12Model12300(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12400":
        agent = RlNnMaxQoeMarginG12Model12400(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12500":
        agent = RlNnMaxQoeMarginG12Model12500(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12600":
        agent = RlNnMaxQoeMarginG12Model12600(env, args)

    elif agentName == "rlNnMaxQoeMarginG12Model12107":
        agent = RlNnMaxQoeMarginG12Model12107(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12207":
        agent = RlNnMaxQoeMarginG12Model12207(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12307":
        agent = RlNnMaxQoeMarginG12Model12307(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12407":
        agent = RlNnMaxQoeMarginG12Model12407(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12507":
        agent = RlNnMaxQoeMarginG12Model12507(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12607":
        agent = RlNnMaxQoeMarginG12Model12607(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12707":
        agent = RlNnMaxQoeMarginG12Model12707(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model12807":
        agent = RlNnMaxQoeMarginG12Model12807(env, args)


    elif agentName == "rlNnMaxQoeMarginG12Model13100":
        agent = RlNnMaxQoeMarginG12Model13100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model13200":
        agent = RlNnMaxQoeMarginG12Model13200(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model13300":
        agent = RlNnMaxQoeMarginG12Model13300(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model13400":
        agent = RlNnMaxQoeMarginG12Model13400(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model13500":
        agent = RlNnMaxQoeMarginG12Model13500(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model13600":
        agent = RlNnMaxQoeMarginG12Model13600(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model11100":
        agent = RlNnMaxQoeMarginG12Model11100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model11200":
        agent = RlNnMaxQoeMarginG12Model11200(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model11300":
        agent = RlNnMaxQoeMarginG12Model11300(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model11400":
        agent = RlNnMaxQoeMarginG12Model11400(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model11500":
        agent = RlNnMaxQoeMarginG12Model11500(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model11600":
        agent = RlNnMaxQoeMarginG12Model11600(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model22100":
        agent = RlNnMaxQoeMarginG12Model22100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model22200":
        agent = RlNnMaxQoeMarginG12Model22200(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model22300":
        agent = RlNnMaxQoeMarginG12Model22300(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model22400":
        agent = RlNnMaxQoeMarginG12Model22400(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model22500":
        agent = RlNnMaxQoeMarginG12Model22500(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model22600":
        agent = RlNnMaxQoeMarginG12Model22600(env, args)

    elif agentName == "rlNnMaxQoeMarginG12Model23100":
        agent = RlNnMaxQoeMarginG12Model23100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model23200":
        agent = RlNnMaxQoeMarginG12Model23200(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model23300":
        agent = RlNnMaxQoeMarginG12Model23300(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model23400":
        agent = RlNnMaxQoeMarginG12Model23400(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model23500":
        agent = RlNnMaxQoeMarginG12Model23500(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model23600":
        agent = RlNnMaxQoeMarginG12Model23600(env, args)

    elif agentName == "rlNnMaxQoeMarginG12Model24107":
        agent = RlNnMaxQoeMarginG12Model24107(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24207":
        agent = RlNnMaxQoeMarginG12Model24207(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24307":
        agent = RlNnMaxQoeMarginG12Model24307(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24407":
        agent = RlNnMaxQoeMarginG12Model24407(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24507":
        agent = RlNnMaxQoeMarginG12Model24507(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24607":
        agent = RlNnMaxQoeMarginG12Model24607(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24707":
        agent = RlNnMaxQoeMarginG12Model24707(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24807":
        agent = RlNnMaxQoeMarginG12Model24807(env, args)

    elif agentName == "rlNnMaxQoeMarginG12Model24110":
        agent = RlNnMaxQoeMarginG12Model24110(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24210":
        agent = RlNnMaxQoeMarginG12Model24210(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24310":
        agent = RlNnMaxQoeMarginG12Model24310(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24410":
        agent = RlNnMaxQoeMarginG12Model24410(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24510":
        agent = RlNnMaxQoeMarginG12Model24510(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24610":
        agent = RlNnMaxQoeMarginG12Model24610(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24710":
        agent = RlNnMaxQoeMarginG12Model24710(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model24810":
        agent = RlNnMaxQoeMarginG12Model24810(env, args)


    elif agentName == "rlNnMaxQoeMarginG12Model32100":
        agent = RlNnMaxQoeMarginG12Model32100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model32200":
        agent = RlNnMaxQoeMarginG12Model32200(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model32300":
        agent = RlNnMaxQoeMarginG12Model32300(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model32400":
        agent = RlNnMaxQoeMarginG12Model32400(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model32500":
        agent = RlNnMaxQoeMarginG12Model32500(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model32600":
        agent = RlNnMaxQoeMarginG12Model32600(env, args)

    elif agentName == "rlNnMaxQoeMarginG12Model33100":
        agent = RlNnMaxQoeMarginG12Model33100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model33200":
        agent = RlNnMaxQoeMarginG12Model33200(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model33300":
        agent = RlNnMaxQoeMarginG12Model33300(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model33400":
        agent = RlNnMaxQoeMarginG12Model33400(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model33500":
        agent = RlNnMaxQoeMarginG12Model33500(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model33600":
        agent = RlNnMaxQoeMarginG12Model33600(env, args)

    elif agentName == "rlNnMaxQoeMarginG12Model34107":
        agent = RlNnMaxQoeMarginG12Model34107(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model34207":
        agent = RlNnMaxQoeMarginG12Model34207(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model34307":
        agent = RlNnMaxQoeMarginG12Model34307(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model34407":
        agent = RlNnMaxQoeMarginG12Model34407(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model34507":
        agent = RlNnMaxQoeMarginG12Model34507(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model34607":
        agent = RlNnMaxQoeMarginG12Model34607(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model34707":
        agent = RlNnMaxQoeMarginG12Model34707(env, args)

    elif agentName == "randomG12":
        agent = RandomG12(env, args)
    elif agentName == "maxRsrqG12":
        agent = MaxRsrqG12(env, args)
    elif agentName == "minUeG12":
        agent = MinUeG12(env, args)

    elif agentName == "rlNnMaxQoeMarginG12Model01010100":
        agent = RlNnMaxQoeMarginG12Model01010100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model01010200":
        agent = RlNnMaxQoeMarginG12Model01010200(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model01010300":
        agent = RlNnMaxQoeMarginG12Model01010300(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model01010400":
        agent = RlNnMaxQoeMarginG12Model01010400(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model01010500":
        agent = RlNnMaxQoeMarginG12Model01010500(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model01010600":
        agent = RlNnMaxQoeMarginG12Model01010600(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model01010700":
        agent = RlNnMaxQoeMarginG12Model01010700(env, args)

    elif agentName == "rlNnMaxQoeMarginG12Model02010100":
        agent = RlNnMaxQoeMarginG12Model02010100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model02010200":
        agent = RlNnMaxQoeMarginG12Model02010200(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model02010300":
        agent = RlNnMaxQoeMarginG12Model02010300(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model02010400":
        agent = RlNnMaxQoeMarginG12Model02010400(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model02010500":
        agent = RlNnMaxQoeMarginG12Model02010500(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model02010600":
        agent = RlNnMaxQoeMarginG12Model02010600(env, args)

    elif agentName == "rlNnMaxQoeMarginG12Model03010100":
        agent = RlNnMaxQoeMarginG12Model03010100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model03010200":
        agent = RlNnMaxQoeMarginG12Model03010200(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model03010300":
        agent = RlNnMaxQoeMarginG12Model03010300(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model03010400":
        agent = RlNnMaxQoeMarginG12Model03010400(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model03010500":
        agent = RlNnMaxQoeMarginG12Model03010500(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model03010600":
        agent = RlNnMaxQoeMarginG12Model03010600(env, args)

    elif agentName == "rlNnMaxQoeMarginG12Model04010100":
        agent = RlNnMaxQoeMarginG12Model04010100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model04010200":
        agent = RlNnMaxQoeMarginG12Model04010200(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model04010300":
        agent = RlNnMaxQoeMarginG12Model04010300(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model04010400":
        agent = RlNnMaxQoeMarginG12Model04010400(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model04010500":
        agent = RlNnMaxQoeMarginG12Model04010500(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model04010600":
        agent = RlNnMaxQoeMarginG12Model04010600(env, args)

    elif agentName == "rlNnMaxQoeMarginG12Model05010100":
        agent = RlNnMaxQoeMarginG12Model05010100(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model05010700":
        agent = RlNnMaxQoeMarginG12Model05010700(env, args)
    elif agentName == "rlNnMaxQoeMarginG12Model10311803":
        agent = RlNnMaxQoeMarginG12Model10311803(env, args)

    agent.run()


def main():

    args = get_args()
    env = get_env(args)
    run_policy(env, args)

if __name__ == "__main__":
    main()
