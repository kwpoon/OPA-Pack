import time
import gym
import numpy as np
from torch import load
from tools_real import getRotationMatrix
import transforms3d
from .Interface import Interface
from .IRcreator import RandomItemCreator, LoadItemCreator, RandomInstanceCreator, RandomCateCreator
from .space import Space
from .cvTools import getConvexHullActions
import random
import threading
import wandb
import copy

rate_ = 20
alpha = 0.1
beta = 0.03
gamma = 1e+7
avoid_threshold = 3
theta = 0.01 #0.03/avoid_threshold



def non_blocking_simulation(interface, finished, id, non_blocking_result):
    succeeded, valid = interface.simulateToQuasistatic(givenId=id,
                                                       linearTol=0.01,
                                                       angularTol=0.01)

    finished[0] = True
    non_blocking_result[0] = [succeeded, valid]


class PackingGame(gym.Env):
    def __init__(self,
                 args
                 ):
        self.args = args
        args = vars(args)
        self.alpha = args['alpha']
        self.beta = args['beta']
        self.theta = args['theta']
        self.resolutionAct = args['resolutionA']
        self.resolutionH = args['resolutionH']
        self.bin_dimension = args['bin_dimension']
        self.scale = args['scale']
        self.objPath = args['objPath']
        self.meshScale = args['meshScale']
        self.shapeDict = args['shapeDict']  # list of 96 meshes
        self.infoDict = args['infoDict']  # Info of 96 types objects
        self.dicPath = load(args['dicPath'])
        self.ZRotNum = args['ZRotNum']
        self.heightMapPre = args['heightMap']
        self.globalView = args['only_simulate_current']
        self.selectedAction = args['selectedAction']
        self.bufferSize = args['bufferSize']
        self.chooseItem = self.bufferSize > 1
        self.simulation = args['simulation']
        self.evaluate = args['evaluate']
        self.maxBatch = args['maxBatch']
        self.heightResolution = args['resolutionZ']
        self.dataSample = args['dataSample']
        self.dataname = args['test_name']
        self.visual = args['visual']
        self.non_blocking = args['non_blocking']
        self.time_limit = args['time_limit']
        self.valina_reward = args['valina_reward']

        self.visual = True

        self.interface = None
        self.item_vec = np.zeros(
            (1000, 9))  # array of (0: object type, 1-3: current position, 4-7: current orientation, 8:int(1))
        self.record_act = np.zeros((1000, 5))  # array of (0: object type, 1: orientation ID, 2-4: current position)
        self.rangeX_A, self.rangeY_A = np.ceil(self.bin_dimension[0:2] / self.resolutionAct).astype(np.int32)
        self.space = Space(self.bin_dimension, self.resolutionAct, self.resolutionH, False, self.ZRotNum,
                           args['shotInfo'], self.scale)

        if self.evaluate and self.dataname is not None:
            self.item_creator = LoadItemCreator(data_name=self.dataname)
        else:
            if self.dataSample == 'category':
                self.item_creator = RandomCateCreator(np.arange(0, len(self.shapeDict.keys())), self.dicPath)
            elif self.dataSample == 'instance':
                self.item_creator = RandomInstanceCreator(np.arange(0, len(self.shapeDict.keys())), self.dicPath)
            else:
                assert self.dataSample == 'pose'
                self.item_creator = RandomItemCreator(np.arange(0, len(self.shapeDict.keys())))

        self.next_item_vec = np.zeros((9))  # list of type id of the next 9 objects

        self.item_idx = 0  # total number of currently packed objects

        self.transformation = []
        DownFaceList, ZRotList = getRotationMatrix(1, self.ZRotNum)
        for d in DownFaceList:
            for z in ZRotList:
                quat = transforms3d.quaternions.mat2quat(np.dot(z, d)[0:3, 0:3])
                self.transformation.append([quat[1], quat[2], quat[3], quat[0]])  # Saved in xyzw
        self.transformation = np.array(self.transformation)

        self.rotNum = self.ZRotNum
        self.act_len = self.selectedAction

        if self.chooseItem:
            self.act_len = self.bufferSize

        if not self.chooseItem:
            self.obs_len = len(self.next_item_vec.reshape(-1))

            if self.selectedAction:
                self.obs_len += self.selectedAction * 5
            else:
                self.obs_len += self.act_len
        else:
            self.obs_len = self.bufferSize

        if self.heightMapPre:
            self.obs_len += self.space.heightmapC.size

        self.obs_len += 50 * 4

        self.observation_space = gym.spaces.Box(low=0.0, high=self.bin_dimension[2],
                                                shape=(self.obs_len,))
        self.action_space = gym.spaces.Discrete(self.act_len)

        self.tolerance = 0  # defalt 0.002

        self.episodeCounter = 0
        self.updatePeriod = 500
        self.trajs = []
        self.orderAction = 0
        self.hierachical = False

        if self.non_blocking:
            self.nullObs = np.zeros((self.obs_len))
            self.finished = [True]
            self.non_blocking_result = [None]
            self.nowTask = False

        #self.relation_forbidden_graph = np.load('relation_forbidden_graph.npy')

    def seed(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]

    def close(self):
        self.interface.close()

    def reset(self, index=None):

        self.space.reset()

        self.episodeCounter = (self.episodeCounter + 1) % self.updatePeriod
        if self.episodeCounter == 0 or self.interface is None:
            if self.interface is not None:
                self.interface.close()
                del self.interface
            self.interface = Interface(bin=self.bin_dimension, foldername=self.objPath, visual=self.visual,
                                       scale=self.scale, simulationScale=self.meshScale, maxBatch=self.maxBatch, )
        else:
            self.interface.reset()
        self.item_creator.reset(index)
        self.packed = []  # list of [(object_type, object_mesh_dir, positionT, orientationT)]
        self.packedId = []  # list of pybullet_obj_ids
        self.next_item_vec[:] = 0
        self.item_idx = 0
        self.item_vec[:] = 0
        self.record_act[:] = 0
        self.id = None
        return self.cur_observation()

    def get_ratio(self):
        totalVolume = 0
        for idx in range(self.item_idx):
            totalVolume += self.infoDict[int(self.item_vec[idx][0])][0]['volume']
        return totalVolume / np.prod(self.bin_dimension)

    '''def get_pressure(self):
        import pybullet as p
        p.stepSimulation()
        accumulated_force = 0.0
        cnt_fragile = 0
        for i in range(self.item_idx):
            if not self.infoDict[int(self.item_vec[i][0])][0]['fragile']:
                continue
            cnt_fragile += 1
            for j in range(self.item_idx):
                contact_points = p.getContactPoints(self.packedId[i], self.packedId[j])
                for contact in contact_points:
                    applied_impulse = contact[9]  # Index 9 corresponds to the applied impulse
                    # Scale by delta time to get the applied force
                    delta_time = 1 / 240.
                    applied_force = applied_impulse / delta_time
                    accumulated_force += applied_force
        if cnt_fragile > 0:
            accumulated_force = accumulated_force / cnt_fragile
        return accumulated_force'''
    
    
    def get_pressure(self):
        import pybullet as p
        p.stepSimulation()
        cnt_fragile = 0
        max_force = 0.0
        for i in range(self.item_idx):
            accumulated_force = 0.0
            if not self.infoDict[int(self.item_vec[i][0])][0]['fragile']:
                continue
            cnt_fragile += 1
            bounds_i = self.interface.get_wraped_AABB(self.packedId[i], inner=False)
            for j in range(self.item_idx):
                if i == j:
                    continue
                weight_j = self.infoDict[int(self.item_vec[j][0])][0]['weight density level'] * \
                        self.infoDict[int(self.item_vec[j][0])][0]['volume']
                bounds_j = self.interface.get_wraped_AABB(self.packedId[j], inner=False)
                if bounds_j[0][2] > bounds_i[1][2] and self.interface.get_trimesh_overlap2d(self.packedId[i], self.packedId[j]):
                    accumulated_force += weight_j
            max_force = max(max_force, accumulated_force)
        return max_force
        '''if max_force == 0:
            return max_force
        else:
            return max_force / cnt_fragile'''

    def get_distance(self):
        average_forb_dis = 0.0
        dis_cnt = 0
        for i in range(self.item_idx):
            for j in range(i + 1, self.item_idx):
                if self.infoDict[int(self.item_vec[j][0])][0]['tag'] in self.infoDict[int(self.item_vec[i][0])][0][
                    'forbidden_list'] \
                        or self.infoDict[int(self.item_vec[i][0])][0]['tag'] in \
                        self.infoDict[int(self.item_vec[j][0])][0]['forbidden_list']:
                    dis_cnt += 1
                    tmp_dis = np.sqrt((( \
                                                   self.interface.get_Wraped_Position_And_Orientation(self.packedId[i],
                                                                                                      inner=False)[0] - \
                                                   self.interface.get_Wraped_Position_And_Orientation(self.packedId[j],
                                                                                                      inner=False)[
                                                       0]) ** 2).sum() + 1e-4)
                    average_forb_dis += tmp_dis
                    if tmp_dis > 0.5:
                        pass
                        '''print('**************************')
                        print('**************************')
                        print('**************************')
                        print('total:', self.item_idx, '(i,j):(', i, j, ') forb_dis:', tmp_dis)
                        print('**************************')
                        print('**************************')
                        print('**************************')'''
        if dis_cnt > 0:
            average_forb_dis = average_forb_dis / dis_cnt
        return average_forb_dis

    def get_surface_distance(self):
        average_forb_dis = 0.0
        dis_cnt = 0
        for i in range(self.item_idx):
            min_dist = 10.0
            dis_cnt += 1
            for j in range(self.item_idx):
                if i == j:
                    continue
                if self.infoDict[int(self.item_vec[j][0])][0]['tag'] in self.infoDict[int(self.item_vec[i][0])][0][
                    'forbidden_list'] \
                        or self.infoDict[int(self.item_vec[i][0])][0]['tag'] in \
                        self.infoDict[int(self.item_vec[j][0])][0]['forbidden_list']:
                    tmp_dis = self.interface.get_trimesh_Surface_Distance(self.packedId[i], self.packedId[j])
                    min_dist = min(min_dist, tmp_dis)
            average_forb_dis += min_dist
        if dis_cnt == 0:
            average_forb_dis = 10.0
        else:
            average_forb_dis = average_forb_dis / dis_cnt
        return average_forb_dis

    def decide_forbid_pair(self, next_item_ID, id): # 1 if the object is packed next to a forbidden object, 0 otherwise
        average_forb_dis = 10.0
        dis_cnt = 0
        for j in range(self.item_idx):
            if int(self.item_vec[j][0]) == next_item_ID:
                continue
            if self.infoDict[int(self.item_vec[j][0])][0]['tag'] in self.infoDict[next_item_ID][0][
                'forbidden_list'] \
                    or self.infoDict[next_item_ID][0]['tag'] in \
                    self.infoDict[int(self.item_vec[j][0])][0]['forbidden_list']:
                tmp_dis = self.interface.get_trimesh_Surface_Distance(id, self.packedId[j])
                if tmp_dis < average_forb_dis:
                    return 1.0
        return 0.0

    def decide_overlap_fragile(self, next_item_ID, id):# No. of fragile object squeezed by the object
        weight_next = self.infoDict[next_item_ID][0]['weight density level'] * self.infoDict[next_item_ID][0][
            'volume']
        #fragile_beneath = 0.0
        sum_weight = 0.0
        for j in range(self.item_idx):
            if int(self.item_vec[j][0]) == next_item_ID:
                continue
            if self.infoDict[int(self.item_vec[j][0])][0]['fragile']:
                weight_j = self.infoDict[int(self.item_vec[j][0])][0]['weight density level'] * \
                           self.infoDict[int(self.item_vec[j][0])][0]['volume']
                if self.interface.get_trimesh_overlap2d(id, self.packedId[j]):
                    sum_weight += weight_j
        return sum_weight

    def get_item_surface_distance(self, next_item_ID, id):
        average_forb_dis = 10.0
        dis_cnt = 0
        for j in range(self.item_idx):
            if int(self.item_vec[j][0]) == next_item_ID:
                continue
            if self.infoDict[int(self.item_vec[j][0])][0]['tag'] in self.infoDict[next_item_ID][0][
                'forbidden_list'] \
                    or self.infoDict[next_item_ID][0]['tag'] in \
                    self.infoDict[int(self.item_vec[j][0])][0]['forbidden_list']:
                dis_cnt += 1
                tmp_dis = self.interface.get_trimesh_Surface_Distance(id, self.packedId[j])
                average_forb_dis = np.minimum(average_forb_dis, tmp_dis)
        if dis_cnt > 0:
            pass
        else:
            average_forb_dis = 10.0
        return average_forb_dis

    def no_unsuitable_contact(self):
        '''import pybullet as p
        p.stepSimulation()
        cnt_all_unsuitable = 0.0
        cnt_success_unsuitable = 0.0
        for i in range(self.item_idx):
            for j in range(i + 1, self.item_idx):
                cnt_all_unsuitable += 1
                if self.infoDict[int(self.item_vec[j][0])][0]['tag'] in self.infoDict[int(self.item_vec[i][0])][0][
                    'forbidden_list'] \
                        or self.infoDict[int(self.item_vec[i][0])][0]['tag'] in \
                        self.infoDict[int(self.item_vec[j][0])][0]['forbidden_list']:
                    contact_points = p.getContactPoints(self.packedId[i], self.packedId[j])
                    tmp_dis = self.interface.get_trimesh_Surface_Distance(self.packedId[i], self.packedId[j])
                    if len(contact_points) > 0 and tmp_dis >= 5.0:
                        print ('len(contact_points)=', len(contact_points), 'tmp_dis', tmp_dis)
                    if tmp_dis < 5.0:
                        pass
                    else:
                        cnt_success_unsuitable += 1
                else:
                    cnt_success_unsuitable += 1
                    #if len(contact_points) > 0:
                    #    pass
                    #else:
                    #    cnt_success_unsuitable += 1
        if cnt_all_unsuitable == 0.0:
            return 1.0
        else:
            return cnt_success_unsuitable / cnt_all_unsuitable'''

        import pybullet as p
        p.stepSimulation()
        cnt_all_unsuitable = 0.0
        cnt_unsuitable = 0.0
        for i in range(self.item_idx):
            for j in range(self.item_idx):
                if i == j:
                    continue
                if self.infoDict[int(self.item_vec[j][0])][0]['tag'] in self.infoDict[int(self.item_vec[i][0])][0][
                    'forbidden_list'] \
                        or self.infoDict[int(self.item_vec[i][0])][0]['tag'] in \
                        self.infoDict[int(self.item_vec[j][0])][0]['forbidden_list']:
                    cnt_all_unsuitable += 1
                    contact_points = p.getContactPoints(self.packedId[i], self.packedId[j])
                    tmp_dis = self.interface.get_trimesh_Surface_Distance(self.packedId[i], self.packedId[j])
                    if len(contact_points) > 0 and tmp_dis >= 3.0:
                        pass
                        #print('len(contact_points)=', len(contact_points), 'tmp_dis', tmp_dis)
                    if tmp_dis < 5.0:
                        return 0.0
                        #return 0.0
                    # if len(contact_points) > 0:
                    #    return 0.0
        #return 1.0
        #print('cnt_success_unsuitable:', cnt_success_unsuitable, 'cnt_all_unsuitable:', cnt_all_unsuitable)
        #if cnt_all_unsuitable == 0.0:
        #    return -1.0
        #else:
        return 1.0
        #return self.item_idx - cnt_unsuitable #/ cnt_all_unsuitable

    def no_unsuitable_contact_overall(self):
        import pybullet as p
        p.stepSimulation()
        cnt_all_unsuitable = 0.0
        cnt_success_unsuitable = 0.0
        for i in range(self.item_idx):
            for j in range(i + 1, self.item_idx):
                if self.infoDict[int(self.item_vec[j][0])][0]['tag'] in self.infoDict[int(self.item_vec[i][0])][0][
                    'forbidden_list'] \
                        or self.infoDict[int(self.item_vec[i][0])][0]['tag'] in \
                        self.infoDict[int(self.item_vec[j][0])][0]['forbidden_list']:
                    contact_points = p.getContactPoints(self.packedId[i], self.packedId[j])
                    tmp_dis = self.interface.get_trimesh_Surface_Distance(self.packedId[i], self.packedId[j])
                    if len(contact_points) > 0 and tmp_dis >= 5.0:
                        pass
                        #print ('len(contact_points)=', len(contact_points), 'tmp_dis', tmp_dis)
                    if tmp_dis < 5.0:
                        return 0.0
                    #if len(contact_points) > 0:
                    #    return 0.0
        return 1.0

    def item_avoid_count(self, next_item_ID, id):
        import pybullet as p
        p.stepSimulation()
        avoid_count = 0
        for j in range(self.item_idx):
            if self.infoDict[int(self.item_vec[j][0])][0]['tag'] in self.infoDict[next_item_ID][0][
                'forbidden_list'] \
                    or self.infoDict[next_item_ID][0]['tag'] in \
                    self.infoDict[int(self.item_vec[j][0])][0]['forbidden_list']:
                avoid_count += 1
        #print ('avoid count:', avoid_count)
        return avoid_count

    def get_item_ratio(self, next_item_ID):
        return self.infoDict[next_item_ID][0]['volume'] / np.prod(self.bin_dimension)

    def gen_next_item_ID(self):
        return self.item_creator.preview(1)[0]

    def get_action_candidates(self, orderAction):
        self.hierachical = True
        self.next_item_ID = self.next_k_item_ID[orderAction]
        self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)
        self.chooseItem = False
        locObservation = self.cur_observation(genItem=False)
        self.chooseItem = True
        self.orderAction = orderAction
        return locObservation

    def get_all_possible_observation(self):
        self.hierachical = True
        self.chooseItem = False
        all_obs = []
        for itemID in self.next_k_item_ID:
            self.next_item_ID = itemID
            self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)
            locObservation = self.cur_observation(genItem=False)
            all_obs.append(locObservation)
        return np.concatenate(all_obs, axis=0)

    def cur_observation(self, genItem=True, draw=False):
        if self.item_idx != 0:
            positions, orientations = self.interface.getAllPositionAndOrientation(inner=False)
            self.item_vec[0:self.item_idx, 1:4] = np.array([positions[0:self.item_idx]])
            self.item_vec[0:self.item_idx, 4:8] = np.array([orientations[0:self.item_idx]])
        if not self.chooseItem:
            if genItem:
                self.next_item_ID = self.gen_next_item_ID()
            self.next_item_vec[0] = self.next_item_ID

            naiveMask = self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID],
                                                         self.selectedAction)

            result = self.next_item_vec.reshape(-1)

            if not self.selectedAction:
                result = np.concatenate((self.next_item_vec.reshape(-1),
                                         naiveMask.reshape(-1)))

            if self.heightMapPre:
                result = np.concatenate((result, self.space.heightmapC.reshape(-1)))
            if self.selectedAction:
                self.candidates = None
                self.candidates = getConvexHullActions(self.space.posZValid, self.space.naiveMask,
                                                       self.heightResolution)
                if self.candidates is not None:
                    if len(self.candidates) > self.selectedAction:
                        # sort with height
                        selectedIndex = np.argsort(self.candidates[:, 3])[0: self.selectedAction]
                        self.candidates = self.candidates[selectedIndex]
                    elif len(self.candidates) < self.selectedAction:
                        dif = self.selectedAction - len(self.candidates)
                        self.candidates = np.concatenate((self.candidates, np.zeros((dif, 5))), axis=0)

                if self.candidates is None:
                    poszFlatten = self.space.posZValid.reshape(-1)
                    selectedIndex = np.argsort(poszFlatten)[0: self.selectedAction]
                    ROT, X, Y = np.unravel_index(selectedIndex, (self.rotNum, self.rangeX_A, self.rangeY_A))
                    H = poszFlatten[selectedIndex]
                    V = self.space.naiveMask.reshape(-1)[selectedIndex]
                    H[:] = self.bin_dimension[-1]
                    self.candidates = np.concatenate([ROT.reshape(-1, 1), X.reshape(-1, 1),
                                                      Y.reshape(-1, 1), H.reshape(-1, 1), V.reshape(-1, 1)], axis=1)
                result = np.concatenate((self.candidates.reshape(-1), result))
        else:
            self.next_k_item_ID = self.item_creator.preview(self.bufferSize)
            result = np.concatenate((np.array(self.next_k_item_ID), self.space.heightmapC.reshape(-1)))

        # Add object properties
        if self.item_idx < 50:
            result = np.concatenate((result, self.record_act[0: 50, 0: 5].reshape(-1)))
        else:
            result = np.concatenate((result, self.record_act[self.item_idx - 50: self.item_idx, 0: 5].reshape(-1)))

        # Add all possible placement candidates into the object selector
        if self.chooseItem:
            for candidate_item_ID in self.next_k_item_ID:
                naiveMask_by_item = self.space.get_possible_position(candidate_item_ID, self.shapeDict[candidate_item_ID],
                                                         self.selectedAction)
                candidates_by_item = getConvexHullActions(self.space.posZValid, self.space.naiveMask,
                                                       self.heightResolution)
                if candidates_by_item is not None:
                    if len(candidates_by_item) > self.selectedAction:
                        # sort with height
                        selectedIndex = np.argsort(candidates_by_item[:, 3])[0: self.selectedAction]
                        candidates_by_item = candidates_by_item[selectedIndex]
                    elif len(candidates_by_item) < self.selectedAction:
                        dif = self.selectedAction - len(candidates_by_item)
                        candidates_by_item = np.concatenate((candidates_by_item, np.zeros((dif, 5))), axis=0)
                if candidates_by_item is None:
                    poszFlatten = copy.deepcopy(self.space.posZValid.reshape(-1))
                    selectedIndex = np.argsort(poszFlatten)[0: self.selectedAction]
                    ROT, X, Y = np.unravel_index(selectedIndex, (self.rotNum, self.rangeX_A, self.rangeY_A))
                    H = poszFlatten[selectedIndex]
                    V = copy.deepcopy(self.space.naiveMask.reshape(-1)[selectedIndex])
                    H[:] = self.bin_dimension[-1]
                    candidates_by_item = np.concatenate([ROT.reshape(-1, 1), X.reshape(-1, 1),
                                                      Y.reshape(-1, 1), H.reshape(-1, 1), V.reshape(-1, 1)], axis=1)
                result = np.concatenate((result, candidates_by_item.reshape(-1)))
        return result

    def action_to_position(self, action):
        rotIdx, lx, ly = self.candidates[action][0:3].astype(np.int)
        return rotIdx, np.round((lx * self.resolutionAct, ly * self.resolutionAct, self.bin_dimension[2]),
                                decimals=6), (lx, ly)

    def prejudge(self, rotIdx, translation, naiveMask):
        extents = self.shapeDict[self.next_item_ID][rotIdx].extents
        if np.round(translation[0] + extents[0] - self.bin_dimension[0], decimals=6) > 0 \
                or np.round(translation[1] + extents[1] - self.bin_dimension[1], decimals=6) > 0:
            return False
        if np.sum(naiveMask) == 0:
            return False
        return True

    # Note the transform between Ra coord and Rh coord
    def step(self, action, direct_rxy=False):
        if self.non_blocking and not self.finished[0]:
            return self.nullObs, 0.0, False, {'Valid': False}

        accumulated_force_before = self.get_pressure()
        forb_dist = self.get_surface_distance()

        if self.non_blocking and self.finished[0] and self.nowTask:
            success, sim_suc = self.non_blocking_result[0]
            self.nowTask = False
            self.non_blocking_result[0] = None

        else:
            if direct_rxy:
                lx, ly = action[1], action[2]
                rotIdx, targetFLB, coordinate = action[0], np.round(
                    (lx * self.resolutionAct, ly * self.resolutionAct, self.bin_dimension[2]), decimals=6), (lx, ly)
            else:
                rotIdx, targetFLB, coordinate = self.action_to_position(action)
            rotation = self.transformation[int(rotIdx)]

            sim_suc = False
            success = self.prejudge(rotIdx, targetFLB, self.space.naiveMask)

            color = np.random.rand(3).tolist() + [1]
            self.id = self.interface.addObject(self.dicPath[self.next_item_ID][0:-4], targetFLB=targetFLB,
                                               rotation=rotation,
                                               linearDamping=0.5, angularDamping=0.5, color=None,
                                               weight_density_level=self.infoDict[self.next_item_ID][0][
                                                   'weight density level'])

            height = self.space.posZmap[rotIdx, coordinate[0], coordinate[1]]
            self.interface.adjustHeight(self.id, height + self.tolerance)

            if success:
                if self.simulation:
                    if self.non_blocking:
                        self.finished[0] = False
                        subProcess = threading.Thread(target=non_blocking_simulation, args=(
                        self.interface, self.finished, self.id, self.non_blocking_result))
                        subProcess.start()
                        self.nowTask = True
                        start_time = time.time()
                        end_time = start_time
                        while end_time - start_time < self.time_limit:
                            end_time = time.time()
                            if self.finished[0]:
                                break
                        if not self.finished[0]:
                            return self.nullObs, 0.0, False, {'Valid': False}
                    else:
                        success, sim_suc = self.interface.simulateToQuasistatic(givenId=self.id,
                                                                                linearTol=0.01,
                                                                                angularTol=0.01)
                else:
                    success, sim_suc = self.interface.simulateHeight(self.id)
        if not self.globalView:
            self.interface.disableObject(self.id)

        bounds = self.interface.get_wraped_AABB(self.id, inner=False)
        positionT, orientationT = self.interface.get_Wraped_Position_And_Orientation(self.id, inner=False)
        self.packed.append([self.next_item_ID, self.dicPath[self.next_item_ID], positionT, orientationT])
        self.packedId.append(self.id)

        if not success:
            if self.globalView and self.evaluate:
                for replayIdx, idNow in enumerate(self.packedId):
                    positionT, orientationT = self.interface.get_Wraped_Position_And_Orientation(idNow, inner=False)
                    self.packed[replayIdx][2] = positionT
                    self.packed[replayIdx][3] = orientationT
            reward = 0.0
            info = {'counter': self.item_idx,
                    'ratio': self.get_ratio(),
                    'Valid': True,
                    }
            wandb.log({'compactness': info['ratio']})
            # wandb.log({'alpha * (gamma div force+gamma)':alpha * gamma/(accumulated_force+gamma)})
            wandb.log({'accumulated_force': accumulated_force_before})
            wandb.log({'forbid dist': forb_dist})
            wandb.log({'forbid acc pair': self.no_unsuitable_contact()})
            wandb.log({'forbid acc overall': self.no_unsuitable_contact_overall()})
            observation = self.cur_observation()
            return observation, reward, True, info

        if sim_suc:
            import pybullet as p
            # Assuming you have a collision between two objects
            if self.globalView:
                self.space.shot_whole()
            else:
                self.space.place_item_trimesh(self.shapeDict[self.next_item_ID][0], (positionT, orientationT),
                                              (bounds, self.next_item_ID))
            self.item_vec[self.item_idx, 0] = self.next_item_ID
            self.item_vec[self.item_idx, -1] = 1
            self.record_act[self.item_idx, 0] = self.next_item_ID
            if direct_rxy:
                lx, ly = action[1], action[2]
                rotIdx, targetFLB, coordinate = action[0], np.round(
                    (lx * self.resolutionAct, ly * self.resolutionAct, self.bin_dimension[2]), decimals=6), (lx, ly)
            else:
                rotIdx, targetFLB, coordinate = self.action_to_position(action)
            coorX, coorY = coordinate
            self.record_act[self.item_idx, 1] = rotIdx
            self.record_act[self.item_idx, 2] = coorX * self.space.stepSize
            self.record_act[self.item_idx, 3] = coorY * self.space.stepSize
            self.record_act[self.item_idx, 4] = self.space.posZmap[rotIdx, coorX, coorY]
            # Compute item compactness
            item_ratio = self.get_item_ratio(self.next_item_ID)
            # Compute new force after placement

            accumulated_force_after = self.get_pressure()
            accumulated_force_item = accumulated_force_after # - accumulated_force_before
            #print ('force_before', accumulated_force_before)
            #print ('force_after', accumulated_force_after)
            avoid_count = self.item_avoid_count(self.next_item_ID, self.id)

            alpha = self.alpha
            beta = self.beta
            if self.valina_reward:
                reward = item_ratio * rate_
                wandb.log({'update reward': reward})
            else:
                reward = (item_ratio * rate_ + \
                         - alpha * self.decide_forbid_pair(self.next_item_ID, self.id) + \
                         - beta * self.decide_overlap_fragile(self.next_item_ID, self.id))
                wandb.log({'update reward': reward})
                wandb.log({'term 1': item_ratio * rate_})
                wandb.log({'term 3': - alpha * self.decide_forbid_pair(self.next_item_ID, self.id)})
                wandb.log({'term 4': - beta * self.decide_overlap_fragile(self.next_item_ID, self.id)})
            self.item_idx += 1
            self.item_creator.update_item_queue(self.orderAction)
            self.item_creator.generate_item()  # add a new box to the list
            observation = self.cur_observation()
            return observation, reward, False, {'Valid': True}
        else:
            # Invalid call
            self.packed.pop()
            self.packedId.pop()
            delId = self.interface.objs.pop()
            self.interface.removeBody(delId)
            self.item_creator.update_item_queue(self.orderAction)
            self.item_creator.generate_item()  # Add a new box to the list
            observation = self.cur_observation()
            return observation, 0.0, False, {'Valid': False}