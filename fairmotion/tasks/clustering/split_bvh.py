# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import asyncio
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool

import os
import tqdm
from fairmotion.data import bvh
from fairmotion.ops import motion as motion_ops
from pathlib import Path

def split_bvh(filepath, time_window, output_folder):
    motion = bvh.load(filepath)
    frames_per_time_window = time_window * motion.fps


    motion_filepath_slice = []

    for num, i in enumerate(
        range(0, motion.num_frames(), int(frames_per_time_window / 2))
    ):
        motion_slice = motion_ops.cut(motion, i, i + frames_per_time_window)
        filepath_slice = os.path.join(
            output_folder,
            Path(filepath).stem + "_" + str(num) + ".bvh",
        )
        motion_filepath_slice.append((motion_slice, filepath_slice))

    with ThreadPool(cpu_count()) as pool:
        pool.map(lambda item:bvh.save(*item), motion_filepath_slice)

    print(filepath, "done!")

def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    files = map(str, Path(args.folder).glob("*.bvh"))
    files = list(files)
    split = partial(split_bvh, time_window=args.time_window, output_folder=args.output_folder)

    with Pool(cpu_count()) as pool:
        pool.map(split, files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split files in a folder to overlapping n second clips"
    )
    parser.add_argument(
        "--time-window", type=int, help="overlapping time window in seconds"
    )
    parser.add_argument("--folder", type=str)
    parser.add_argument("--output-folder", type=str)
    args = parser.parse_args("""--time-window 2 --folder data\\output --output-folder data\\clustering""".split(" "))
    main(args)