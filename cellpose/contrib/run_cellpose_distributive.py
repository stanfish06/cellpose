import getpass
import logging
import os
import pathlib

import dask_jobqueue
import distributed
from cellpose.contrib.distributed_segmentation import (
    distributed_eval,
    wrap_folder_of_tiffs,
)

logging.disable(logging.CRITICAL)

# default cellpose sam
model_kwargs = {"gpu": True}
eval_kwargs = {"z_axis": 0, "do_3D": True}
cluster_kwargs = {
    "cores": 2,
    "min_workers": 1,
    "max_workers": 16,
    "walltime": "1:00:00",
    "queue": "gpu",
    "job_extra_directives": [
        "--gres=gpu:1",
    ],
}
# my file names are usually stitched_p0000_w0000_t0000.tif
input_imgs = "./test/*.tif"
index_pattern = r"_(p)(\d+)_(w)(\d+)_(t)(\d+)"
output_path = "./test"


class SlurmCluster(dask_jobqueue.SLURMCluster):
    def __init__(
        self,
        cores,
        min_workers,
        max_workers,
        config={},
        persist_config=False,
        local_directory=f"/scratch/{getpass.getuser()}/",
        job_script_prologue=[],
        **kwargs,
    ):
        # store all args in case needed later
        self.locals_store = {**locals()}

        # config
        self.persist_config = persist_config
        config_defaults = {
            "temporary-directory": local_directory,
            "distributed.comm.timeouts.connect": "180s",
            "distributed.comm.timeouts.tcp": "360s",
        }
        config = {**config_defaults, **config}

        # set log directories
        if "log_directory" not in kwargs:
            log_dir = f"{os.getcwd()}/dask_worker_logs_{os.getpid()}/"
            pathlib.Path(log_dir).mkdir(parents=False, exist_ok=True)
            kwargs["log_directory"] = log_dir

        # construct
        super().__init__(
            processes=1,
            cores=cores,
            memory=str(8 * cores) + "GB",
            job_script_prologue=job_script_prologue,
            **kwargs,
        )
        self.client = distributed.Client(self)
        print("Cluster dashboard link: ", self.dashboard_link)

        # set adaptive cluster bounds
        self.adapt_cluster(min_workers, max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()
        super().__exit__(exc_type, exc_value, traceback)

    def adapt_cluster(self, min_workers, max_workers):
        _ = self.adapt(
            minimum_jobs=min_workers,
            maximum_jobs=max_workers,
            interval="10s",
            wait_count=6,
        )

    def change_worker_attributes(
        self,
        min_workers,
        max_workers,
        **kwargs,
    ):
        """WARNING: this function is dangerous if you don't know what
        you're doing. Don't call this unless you know exactly what
        this does."""
        self.scale(0)
        for k, v in kwargs.items():
            self.new_spec["options"][k] = v
        self.adapt_cluster(min_workers, max_workers)


def main():
    data_zarr = wrap_folder_of_tiffs(input_imgs, block_index_pattern=index_pattern)
    cluster = SlurmCluster(**cluster_kwargs)
    segments, boxes = distributed_eval(
        input_zarr=data_zarr,
        blocksize=(256, 256, 256),
        write_path=str(output_path),
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        cluster=cluster,
    )
    print("done")


if __name__ == "__main__":
    main()
