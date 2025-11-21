import os
import pickle
import subprocess
from pathlib import Path

from cellpose.contrib.distributed_segmentation import (
    SlurmCluster,
    distributed_eval,
    janeliaLSFCluster,
    numpy_array_to_zarr,
)
from tifffile import imread


def main():
    ## PARAMETERS
    cluster_kwargs = {
        "job_cpu": 2,  # number of CPUs per GPU worker
        "ncpus": 1,  # threads requested per GPU worker
        "min_workers": 1,  # min number of workers based on expected workload
        "max_workers": 5,  # max number of workers based on expected workload
        "walltime": "3:00:00",  # available runtime for each GPU worker for cluster scheduler (Slurm, LSF)
        "queue": "spgpu",
        "local_directory": "/home/zyyu/dask-dump",
        "log_directory": "/home/zyyu/dask-dump",
        "job_extra_directives": [
            "--gres=gpu:1",
        ],
    }
    # * Ask your cluster support staff for assistance
    # match stitched_p0000_w0000_t0000.tif
    dataDir = Path(os.getcwd())
    files = []
    nucChannel = 0
    for filename in Path(dataDir).rglob("*_p*_w%.4d_t0000.tif" % nucChannel):
        filepath = os.path.split(filename)[0]
        if filepath == str(dataDir):
            files.append(filename)

    # Cellpose parameters
    model_kwargs = {"gpu": True}
    eval_kwargs = {"z_axis": 0, "do_3D": True, "anisotropy": 5.0}

    ## EVALUATION
    # Guess cluster type by checking for cluster submission commands
    if subprocess.getstatusoutput("sbatch -h")[0] == 0:
        print("Slurm sbatch command detected -> use SlurmCluster")
        cluster = SlurmCluster(**cluster_kwargs)
    elif subprocess.getstatusoutput("bsub -h")[0] == 0:
        print("LSF bsub command detected -> use janeliaLSFCLuster")
        cluster = janeliaLSFCluster(**cluster_kwargs)
    else:
        cluster = None
        ## Note in case you want to test without a cluster scheduler use:
        # from cellpose.contrib.distributed_segmentation import myLocalCluster
        # cluster = myLocalCluster(**{
        #    'n_workers': 1,            # if you only have 1 gpu, then 1 worker is the right choice
        #    'ncpus': 8,
        #    'memory_limit':'64GB',
        #    'threads_per_worker':1,
        # })

    if cluster is None:
        raise Exception(
            "Neither SLURM nor LFS cluster detected. "
            "Currently, this script only supports SLURM or LSF cluster scheduler. "
            "You have two options:"
            "\n * Either use `distributed_eval` without the `cluster` but with the `cluster_kwargs` argument to start a local cluster on your machine"
            "\n * or raise a feature request at https://github.com/MouseLand/cellpose/issues."
        )

    for f in files:
        print(f"process {f.name}")
        # Compute node-accessible directory for input zarr dataset and outputs
        output_dir = Path() / f"{f.name}_outputs"
        input_zarr_path = output_dir / f"{f.name}_input.zarr"
        output_zarr_path = output_dir / f"{f.name}_segmentation.zarr"
        output_bbox_pkl = output_dir / f"{f.name}_bboxes.pkl"

        ## DATA PREPARATION
        data_numpy = imread(f)
        data_zarr = numpy_array_to_zarr(
            input_zarr_path, data_numpy, chunks=(256, 256, 256)
        )

        # Start computation
        _, boxes = distributed_eval(
            input_zarr=data_zarr,
            blocksize=(256, 256, 256),
            write_path=str(output_zarr_path),
            model_kwargs=model_kwargs,
            eval_kwargs=eval_kwargs,
            cluster=cluster,
        )

        # Save bounding boxes on disk
        with open(output_bbox_pkl, "wb") as f:
            pickle.dump(boxes, f)

        print(f"Segmentation saved in {str(output_zarr_path)}")
        print(f"Object bounding boxes saved in {str(output_bbox_pkl)}")


if __name__ == "__main__":
    main()
