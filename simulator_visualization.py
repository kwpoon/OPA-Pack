import pybullet as p
import trimesh
import numpy as np
import os
import argparse
import time
import shutil
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--visual', action='store_true', help='Visualize the simulation')
parser.add_argument('--dataset', type=str, default='oop_cook', help='Dataset name')
parser.add_argument('--case_id', type=int, default=0, help='Case ID')
parser.add_argument('--fragile', action='store_true', help='Show Fragile Objects')
args = parser.parse_args()
args.visual = True
args.fragile = True
args.pair = False



p.connect(p.GUI)

bin_dimension = [0.32, 0.32, 0.30]
scale = [1, 1, 1]
shift = [0, 0, 0]
thick = 1
bin_color=[0.1, 0.1, 0.1, 0.1]

p.setGravity(0, 0, -9.8)
p.setPhysicsEngineParameter(constraintSolverType=p.CONSTRAINT_SOLVER_LCP_PGS, globalCFM=0.0001,
                            numSolverIterations=10)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)


def makeBox(bin_dimension=bin_dimension, color = bin_color):
    box = []

    left = trimesh.primitives.Box(extents=np.array([thick, bin_dimension[1] + thick * 2, bin_dimension[2]]))
    left.apply_translation(np.array([-thick / 2, bin_dimension[1] / 2, bin_dimension[2] / 2]))
    left.visual.face_colors = color

    right = trimesh.primitives.Box(extents=np.array([thick, bin_dimension[1] + thick * 2, bin_dimension[2]]))
    right.apply_translation(np.array([bin_dimension[0] + thick / 2, bin_dimension[1] / 2, bin_dimension[2] / 2]))
    right.visual.face_colors = color

    front = trimesh.primitives.Box(extents=np.array([bin_dimension[0], thick, bin_dimension[2]]))
    front.apply_translation(np.array([bin_dimension[0] / 2, bin_dimension[1] + thick / 2, bin_dimension[2] / 2]))
    front.visual.face_colors = color

    back = trimesh.primitives.Box(extents=np.array([bin_dimension[0], thick, bin_dimension[2]]))
    back.apply_translation(np.array([bin_dimension[0] / 2, -thick / 2, bin_dimension[2] / 2]))
    back.visual.face_colors = color

    bottom = trimesh.primitives.Box(extents=np.array([bin_dimension[0] + thick * 2, bin_dimension[1] + thick * 2, thick]))
    bottom.apply_translation(np.array([bin_dimension[0] / 2, bin_dimension[1] / 2, -thick / 2]))
    bottom.visual.face_colors = color

    box.append(bottom)
    box.append(left)
    box.append(right)
    box.append(front)
    box.append(back)

    return box

def get_trimesh_Position_And_Orientation(id, mesh, inner=True, getPosBase=False):

        positionBase, orientationT = p.getBasePositionAndOrientation(id)
        mat = p.getMatrixFromQuaternion(orientationT)
        mesh.apply_transform(extendMat(np.array(mat).reshape((3, 3)), positionBase))
        bounds = mesh.bounds

        if not inner:
            bounds = bounds / self.defaultScale

        positionFLB = bounds[0]
        returnList = [np.array(positionFLB), np.array(orientationT)]
        if getPosBase:
            returnList.append(np.array(positionBase))

        return returnList

def visualize_results(vis_basedir = 'result_save_dataset_'+args.dataset):

    box = makeBox(bin_dimension=bin_dimension)

    for index, side in enumerate(box):
        if index == 0:
            repeat = 5
        else:
            repeat = 1

        boxPath = os.path.join('dataset', args.dataset, 'box_32.0_32.0_30.0', 'Box' + str(index) + '.obj')
        if not os.path.exists(boxPath):
            side.export(boxPath)

        if args.visual:
            visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                  fileName=boxPath,
                                                  rgbaColor=bin_color,
                                                  specularColor=[0.4, .4, 0],
                                                  visualFramePosition=shift,
                                                  meshScale=0.01
                                                  )

        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                    fileName=boxPath,
                                                    collisionFramePosition=shift,
                                                    flags=1,
                                                    )

        for _ in range(repeat):
            if args.visual:
                boxID = p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=collision_shape_id,
                                          baseVisualShapeIndex=visual_shape_id,
                                          useMaximalCoordinates=True)
            else:
                boxID = p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=collision_shape_id,
                                          useMaximalCoordinates=True)

            p.changeDynamics(boxID, -1,
                             contactProcessingThreshold=0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    target = [bin_dimension[0] / 2, bin_dimension[1] / 2, bin_dimension[2] / 2]
    yaw=45.0; pitch=-30.0
    scale = 100.0
    p.resetDebugVisualizerCamera(cameraDistance=bin_dimension[0]*2*scale,
                                cameraYaw=yaw,
                                cameraPitch=pitch,
                                cameraTargetPosition=np.array(target)*100.0)

    pack_result = np.load(os.path.join(vis_basedir, 'result_case_'+str(args.case_id)+'.npy'), allow_pickle=True).item()
    all_names = []
    for id_str in pack_result.keys():
        id = str(id_str)
        name = pack_result[id_str]['name']
        pose = pack_result[id_str]['pose']
        orien = pack_result[id_str]['orien']
        print (pack_result[id_str])
        file_basename = name[:-2]
        all_names.append(file_basename)
        pose_id = int(name[-1])
        objPath_colorful = os.path.join('dataset', 'textured_mesh_pose0', args.dataset, file_basename, 'mesh.obj')
        objPath_colorless = os.path.join('dataset', args.dataset, 'shape_vhacd', name[0:-2] + '_processed' + name[-2::] + '.obj')
        mesh_colorful = trimesh.load(objPath_colorful)
        mesh_colorless = trimesh.load(objPath_colorless)
        read_poses = torch.load(os.path.join('dataset', 'stable_poses', args.dataset, file_basename + '.pt'))
        combined_mat = np.dot(read_poses[pose_id], np.linalg.inv(read_poses[0]))
        mesh_colorful = mesh_colorful.apply_transform(combined_mat)
        trans_offset = mesh_colorless.bounds[0] - mesh_colorful.bounds[0]
        mesh_colorful = mesh_colorful.apply_translation(trans_offset)
        if not os.path.isdir('tmp_' + name):
            os.makedirs('tmp_' + name)
        mesh_colorful = mesh_colorful.apply_scale(scale)
        mesh_colorless = mesh_colorless.apply_scale(scale)
        mesh_colorful.export(os.path.join('tmp_' + name, 'colored.obj'))
        mesh_colorless.export(os.path.join('tmp_' + name, 'mesh.obj'))
        if os.path.getsize(os.path.join('tmp_' + name, 'colored.obj')) > 2000000:
            mesh_name = os.path.join('tmp_' + name, 'mesh.obj')
        else:
            mesh_name = os.path.join('tmp_' + name, 'colored.obj')
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                              fileName=mesh_name,
                                              meshScale=1)
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                    fileName=os.path.join('tmp_' + name, 'mesh.obj'),
                                                    collisionFramePosition=[0.0, 0.0, 0.0],
                                                    meshScale=1)
        id = p.createMultiBody(baseMass=mesh_colorless.volume*1,
                               basePosition=[-100, -100, -100],
                               baseOrientation=[1,0,0,0],
                               baseCollisionShapeIndex=collision_shape_id,
                               baseVisualShapeIndex=visual_shape_id,
                               useMaximalCoordinates=True)
        p.resetBasePositionAndOrientation(id, pose, orien)
        pack_result[id_str]['new_id'] = id
        linearDamping = 0.1
        angularDamping = 0.1

        shutil.rmtree('tmp_' + name)
        # change dynamics
        p.changeDynamics(id, -1, linearDamping=linearDamping, angularDamping=angularDamping, mass=0)
    import pdb
    pdb.set_trace()
    if args.fragile:
        for id_str in pack_result.keys():
            id = pack_result[id_str]['new_id']
            name = pack_result[id_str]['name']
            print (pack_result[id_str])
            file_basename = name[:-2]
            pose_id = int(name[-1])
            property_annot = np.load(os.path.join('dataset', args.dataset, 'shape_info', name[0:-2] + '_manifold.npy'),
                                     allow_pickle=True).item()
            print(property_annot)
            if property_annot['fragile'].lower() == 'no':
                # change visual color
                p.changeVisualShape(id, -1, rgbaColor=[0.1, 0.1, 0.1, 0.5])

    elif args.pair:
        dict_pair_list = {}
        forbid_dict = np.load(os.path.join('dataset', args.dataset, 'dict_obj_forbid_list_'+args.dataset[4::]+'.npy'), allow_pickle=True).item()
        for id_str in pack_result.keys():
            id = pack_result[id_str]['new_id']
            name = pack_result[id_str]['name']
            print (pack_result[id_str])
            file_basename = name[:-2]
            pose_id = int(name[-1])
            dict_pair_list[file_basename] = {}
            dict_pair_list[file_basename]['id'] = id
            dict_pair_list[file_basename]['forbid_names'] = []
            for file_basename2 in all_names:
                if file_basename2 in forbid_dict[file_basename] or file_basename in forbid_dict[file_basename2]:
                    dict_pair_list[file_basename]['forbid_names'].append(file_basename2)


        list_pair_distance = []
        for file_basename1 in dict_pair_list.keys():
            id1 = dict_pair_list[file_basename1]['id']
            for file_basename2 in dict_pair_list[file_basename1]['forbid_names']:
                id2 = dict_pair_list[file_basename2]['id']
                distance = np.linalg.norm(np.array(p.getBasePositionAndOrientation(id1)[0]) - np.array(p.getBasePositionAndOrientation(id2)[0]))
                list_pair_distance.append((id1, id2, distance))
        list_pair_distance.sort(key=lambda x: x[2])
        print ('total pairs:', len(list_pair_distance))

        show_pair_id = len(list_pair_distance)-1
        id1, id2, _ = list_pair_distance[show_pair_id]

        for basename in dict_pair_list.keys():
            if dict_pair_list[basename]['id'] != id1 and dict_pair_list[basename]['id'] != id2:
                p.changeVisualShape(dict_pair_list[basename]['id'], -1, rgbaColor=[0.1, 0.1, 0.1, 0.5])
            else:
                print (basename)

    import pdb
    pdb.set_trace()
    for _ in range(240*100):
        p.stepSimulation()

args.valina = True
if args.valina:
    visualize_results('result_valina_save_dataset_'+args.dataset)
else:
    visualize_results('result_save_dataset_'+args.dataset)
