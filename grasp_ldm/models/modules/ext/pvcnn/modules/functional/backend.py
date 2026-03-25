import os

_src_path = os.path.dirname(os.path.abspath(__file__))

try:
    from torch.utils.cpp_extension import load

    _backend = load(
        name="_pvcnn_backend",
        extra_cflags=["-O3", "-std=c++17"],
        sources=[
            os.path.join(_src_path, "src", f)
            for f in [
                "ball_query/ball_query.cpp",
                "ball_query/ball_query.cu",
                "grouping/grouping.cpp",
                "grouping/grouping.cu",
                "interpolate/neighbor_interpolate.cpp",
                "interpolate/neighbor_interpolate.cu",
                "interpolate/trilinear_devox.cpp",
                "interpolate/trilinear_devox.cu",
                "sampling/sampling.cpp",
                "sampling/sampling.cu",
                "voxelization/vox.cpp",
                "voxelization/vox.cu",
                "bindings.cpp",
            ]
        ],
    )
except (OSError, ImportError) as e:
    import warnings

    warnings.warn(
        f"PVCNN CUDA backend could not be compiled: {e}. "
        "PVCNN-based models will not work. "
        "Set CUDA_HOME to your CUDA installation to enable."
    )
    _backend = None

__all__ = ["_backend"]
