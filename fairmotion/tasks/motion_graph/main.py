# Copyright (c) Facebook, Inc. and its affiliates.

'''
python test_motion_graph.py --v-up-env y --length 30 --num-files 5 --motion-folder XXX --output-bvh-folder YYY
python fairmotion/tasks/motion_graph/main.py --v-up-env y --length 30 --num-files 5 --motion-folder data\split --output-bvh-folder data\output
'''

import argparse
import pickle
import gzip
import logging
from multiprocessing import cpu_count
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from fairmotion.data import bvh
from fairmotion.core import velocity
from fairmotion.tasks.motion_graph import motion_graph as graph
from fairmotion.utils import utils
from fairmotion.ops import conversions, motion as motion_ops

import os

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser(
        description="Motion graph construction and exploration"
    )
    parser.add_argument(
        "--motion-files", action="append", default=[],
        help="Motion Files")
    parser.add_argument(
        "--motion-folder", action="append", default=[],
        help="Folder that contains motion files"
    )
    parser.add_argument(
        "--output-bvh-folder",
        type=str,
        required=True,
        help="Resulting motion stored as bvh",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        required=True,
        help="Number of files to generate",
    )
    parser.add_argument(
        "--length",
        type=float,
        required=True,
        help="Number of files to generate",
    )

    w_ee_pos = 10.0 # 1.0
    w_ee_vel = 0.1 # 0.01
    w_root_pos = 10.0

    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--v-up-skel", type=str, default="y")
    parser.add_argument("--v-face-skel", type=str, default="z")
    parser.add_argument("--v-up-env", type=str, default="y")
    parser.add_argument("--scale", type=float, default=1.0 * 1e-2)
    parser.add_argument("--base-length", type=float, default=1.0) # figure out it
    parser.add_argument("--stride-length", type=float, default=1.0) # figure out it
    parser.add_argument("--blend-length", type=float, default=0.5 * 1.0) # figure out it
    parser.add_argument("--compare-length", type=float, default=0.2) # figure out it
    parser.add_argument("--diff-threshold", type=float, default=5.0)
    parser.add_argument("--w-joint-pos", type=float, default=1.0)
    parser.add_argument("--w-joint-vel", type=float, default=0.01)
    parser.add_argument("--w-root-pos", type=float, default=w_root_pos)
    parser.add_argument("--w-root-vel", type=float, default=0.01)
    parser.add_argument("--w-ee-pos", type=float, default=w_ee_pos)
    parser.add_argument("--w-ee-vel", type=float, default=w_ee_vel)
    parser.add_argument("--w-trajectory", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-comparison", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=10)

    # args = parser.parse_args("""--verbose --fps 24 --length 30 --num-files 5 --motion-folder data\\bvh_choremaster_split --output-bvh-folder data\\output""".split(" "))
    # args = parser.parse_args("""--verbose --length 50 --num-files 5 --motion-folder data\\split --output-bvh-folder data\\output""".split(" "))

    args = parser.parse_args("""--verbose --fps 24 --length 30 --num-files 5 --motion-folder data\\temp_split --output-bvh-folder data\\output""".split(" "))

    # Load motions
    motion_files = args.motion_files
    if len(args.motion_folder) > 0:
        for d in args.motion_folder:
            motion_files += utils.files_in_dir(d, ext="bvh")


    motion_files = sorted(motion_files)[:]

    if args.verbose:
        print("-----------Motion Files-----------")
        print(motion_files)
        print("----------------------------------")

    def dump(motions, dir):
        import gzip
        import pickle
        def zdump(item):
            idx, motion = item
            filename = f"{dir}\\motion-{idx}.zip"
            with gzip.open(filename, "wb") as f:
                pickle.dump(motion, f)
        from multiprocessing.pool import ThreadPool
        with ThreadPool(cpu_count()) as pool:
            pool.map(zdump, enumerate(motions))

    def load(dir):
        import gzip
        import pickle

        def zload(filename):
            with gzip.open(filename, "rb") as f:
                return pickle.load(f) 
        
        from multiprocessing.pool import ThreadPool
        from pathlib import Path
        import random
        with ThreadPool(cpu_count()) as pool:
            motions_paths = map(str, Path(dir).glob("*.zip"))
            motions_paths = sorted(motions_paths)
            # random.shuffle(motions_paths)
            motions_paths = motions_paths
            return pool.map(zload, motions_paths)

    INIT_MOTION = True
    MOTION_DUMP_DIR = f".\\data\\\dumped_motions"
    
    if INIT_MOTION:
        print("init motions...")
        motions = bvh.load_parallel(
            motion_files,
            scale=args.scale,
            v_up_skel=utils.str_to_axis(args.v_up_skel),
            v_face_skel=utils.str_to_axis(args.v_face_skel),
            v_up_env=utils.str_to_axis(args.v_up_env),
        )
        print("dump motions...")
        dump(motions, MOTION_DUMP_DIR)
        print("done!")
    else:
        print("load motions...")
        motions = load(MOTION_DUMP_DIR)
        print("done!")

    num_joints = np.argmax(np.bincount(np.array([ motion.skel.num_joints() for motion in motions])))
    motions = list(filter(lambda motion: len(motion.poses) // (motion.fps // args.fps) > 1 and len(motion.poses[0].data) == num_joints, motions))

    skel = motions[0].skel
    motions_with_velocity = []
    
    INIT_MOTION_V = True
    MOTION_V_DUMP_DIR = f".\\data\dumped_motions_with_velocity"
    if INIT_MOTION_V:
        for motion in tqdm(motions):
            motion.set_skeleton(skel)
            motion_ops.resample(motion, args.fps)
            motions_with_velocity.append(
                velocity.MotionWithVelocity.from_motion(motion)
            )

        logging.info(f"Loaded {len(motions_with_velocity)} files")
        dump(motions_with_velocity, MOTION_V_DUMP_DIR)
    else:
        motions_with_velocity = load(MOTION_V_DUMP_DIR)


    INIT_GRAPH = True
    if INIT_GRAPH:
        ''' 
        Construct Motion Graph
        We assume all motions have the same
            skeleton hierarchy
            fps
        '''
        mg = graph.MotionGraph(
            motions=motions_with_velocity,
            motion_files=motion_files,
            skel=skel,
            fps=args.fps,
            base_length=args.base_length,
            stride_length=args.stride_length,
            compare_length=args.compare_length,
            verbose=True,
        )
        mg.construct(
            w_joints=None,
            w_joint_pos=args.w_joint_pos,
            w_joint_vel=args.w_joint_vel,
            w_root_pos=args.w_root_pos,
            w_root_vel=args.w_root_vel,
            w_ee_pos=args.w_ee_pos,
            w_ee_vel=args.w_ee_vel,
            w_trajectory=args.w_trajectory,
            diff_threshold=args.diff_threshold,
            num_workers=args.num_workers,
        )


        with gzip.open("temp_motion_graph.gzip", "wb") as f:
            pickle.dump(mg, f)
    else:
        with gzip.open("temp_motion_graph.gzip", "rb") as f:
            mg = pickle.load(f)
    
    # removing_linear_edges : List[Tuple[int, int]] = [(f,t) for f, t in mg.graph.edges if t - f == 2]
    # list(map(lambda edge:mg.graph.remove_edge(*edge), removing_linear_edges))
    print("Nodes %d, Edges %d"%(mg.graph.number_of_nodes(), mg.graph.number_of_edges()))


    def calc_weight(w_joint_pos, w_joint_vel, w_root_pos, w_root_vel, w_ee_pos, w_ee_vel, diff_pos, diff_vel, diff_root_pos, diff_root_vel, diff_ee_pos, diff_ee_vel, **kwargs):
        weights = np.array([w_joint_pos, w_joint_vel, w_root_pos, w_root_vel, w_ee_pos, w_ee_vel])
        value = np.array([diff_pos, diff_vel, diff_root_pos, diff_root_vel, diff_ee_pos, diff_ee_vel])
        return np.dot(weights, value)

    class RemovingEdge():
        def __init__(self, graph) -> None:
            self.graph = graph
            self.remove_plan = []

        def __call__(self, fromto):
            self.remove_plan.append(fromto)

        def commit(self):
            for fromto in (self.remove_plan.pop() for _ in range(len(self.remove_plan))):
                self.graph.remove_edge(*fromto)

    weights_args = dict(w_joint_pos=1.0, w_joint_vel=0.1, w_root_pos=10.0, w_root_vel=0.01, w_ee_pos=10.0, w_ee_vel=1.0)
    removeing_edge = RemovingEdge(mg.graph)
    diff_threhold = 5.0
    for fromto, edge in ((edge, mg.graph.edges[edge]) for edge in mg.graph.edges):
        new_weights = calc_weight(**weights_args, **edge)
        if new_weights > diff_threhold:
            removeing_edge(fromto)
        else:
            edge.update(weights=new_weights)
    
    removeing_edge.commit()

    print("Nodes %d, Edges %d"%(mg.graph.number_of_nodes(), mg.graph.number_of_edges()))

    mg.reduce(method="scc")

    print("Nodes %d, Edges %d"%(mg.graph.number_of_nodes(), mg.graph.number_of_edges()))
    
    cnt = 0
    visit_weights = {}
    visit_discount_factor = 0.1
    nodes = list(mg.graph.nodes)
    for n in nodes:
        visit_weights[n] = 1.0
    for cnt in tqdm(range(args.num_files)):
        m, _ = mg.create_random_motion(
            length=args.length, 
            blend_length=args.blend_length,
            start_node=None, 
            visit_weights=visit_weights,
            visit_discount_factor=visit_discount_factor)
        bvh.save(m, filename=os.path.join(args.output_bvh_folder, "%03d.bvh"%cnt))
        cnt += 1
        print('\r%d/%d completed'%(cnt, args.num_files))
