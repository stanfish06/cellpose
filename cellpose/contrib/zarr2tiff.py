import os
from pathlib import Path

import zarr
from tifffile import imwrite


def main():
    dataDir = Path(os.getcwd())
    files = []
    nucChannel = 0
    for filename in Path(dataDir).rglob("*_p*_w%.4d_t0000.tif" % nucChannel):
        filepath = os.path.split(filename)[0]
        if filepath == str(dataDir):
            files.append(filename)
    for f in files:
        print(f"process {f.name}")
        segment_path = Path() / f"{f.name}_outputs" / f"{f.name}_segmentation.zarr"
        data = zarr.open(segment_path.__str__(), mode="r")
        segment = data[:]
        imwrite(
            f"{f.name.split('.')[0]}_cp_masks.tif",
            segment,
            metadata={"spacing": 1.0, "unit": "um", "axes": "ZYX"},
        )


if __name__ == "__main__":
    main()
