from functools import reduce
from pathlib import Path
from typing import Iterable, List, Tuple


rootdir = r"D:\git-repo\fairmotion\data\bvh_choremaster"

def merge(dance_pattern, paths:List[Path]):
    files = map(open, paths)
    files = map(list, files)
    files = map(lambda lines:(lines[:319], lines[321:]), files)
    def _merge(head_anims:Iterable[Tuple[List[str], List[str]]]) -> str:
        frames_info = lambda frames: "\n".join([
                f"Frames: {frames}",
                "Frame Time: 0.041667",
                ""
            ])

        all_anims : List[str] = []
        frames = 0

        for head, anim in head_anims:
            frames += len(anim)
            all_anims.extend(anim)
        
        return "".join([*head, *frames_info(frames), *all_anims])
    
    return _merge(files)
    


if __name__ == "__main__":
    dance_patterns = set(path.stem.split("@")[0] for path in Path(rootdir).glob("*.bvh"))
    for dance_pattern in dance_patterns:
        merged_dance = merge(dance_pattern, sorted(Path(rootdir).glob(f"{dance_pattern}@*")))
        with open(f"{dance_pattern}.bvh", "w") as f:
            f.write(merged_dance)
        