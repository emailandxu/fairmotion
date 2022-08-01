from itertools import chain
from typing import Tuple
from fairmotion.data import bvh, asfamc
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_motion(asffile, amcfile):
    return asfamc.load(
                file=asffile, motion=amcfile
            )
        
def save_motion(asf_amc:Tuple[Path, Path]):
    asf, amc = asf_amc
    motion = load_motion(asf.absolute(), amc.absolute())
    bvh.save(motion, f"data\output\{amc.stem}.bvh")


def iter_all_asfamc_tuple(all_asfamc_path=r"data\all_asfamc\subjects"):

    subjects = Path(all_asfamc_path).iterdir()

    def make_asf_amc_pair(subject:Path):
        asfs = list(subject.glob("*.asf"))
        assert len(asfs) == 1
        amcs = list(subject.glob("*.amc"))
        return zip(asfs*len(amcs), amcs)

    subject_seprated_pair = map(make_asf_amc_pair, subjects)
    return tuple(chain(*subject_seprated_pair))

if __name__ == "__main__":

    with Pool(cpu_count()) as pool:
        pool.map(save_motion, tqdm(iter_all_asfamc_tuple()))