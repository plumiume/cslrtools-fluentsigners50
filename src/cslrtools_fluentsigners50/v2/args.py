from pathlib import Path
from clipar import namespace
from cslrtools.dataset.v2.torch.plugins import Info

@namespace
class FS50DatasetArgs:
    """Arguments for configuring the FluentSigners50 dataset."""

    base_root: Path
    "Path to the base root directory containing the FS50 dataset."
    processed_root: Path
    "Path to the processed root directory containing annotations and processed files."
    export_path: Path
    "Path to export the dataset in Zarr format."

    original_video_key: str = 'original'
    "Key name for the original video files."
    use_original_video: bool = False
    "Whether to use the original video files from the base root directory."
    enable_cache: bool = True
    "Whether to enable caching for data loaders."

    extra_videos: list[str] = []
    "List of extra video file names to load from the processed root directory."
    landmarks: list[str] = []
    "List of landmark file names to load from the processed root directory."
    self_connection_files: list[str] = []
    "List of self-connection file names to load from the processed root directory."
    other_connection_files: list[str] = []
    "List of other-connection file names to load from the processed root directory."

# if TYPE_CHECKING:
#     from .main import FS50Dataset
# else:
#     FS50Dataset = tuple

def fs50_dataset_from_args(args: FS50DatasetArgs.T):
    """Create an FS50Dataset instance from the provided arguments and export it to Zarr format.

    Args:
        args (FS50DatasetArgs.T): The arguments for configuring the FS50 dataset.

    """

    from cslrtools.dataset.v2.torch.dataset import dataset_to_zarr
    from .main import FS50Dataset
    fs50 = FS50Dataset(
        base_root=args.base_root,
        processed_root=args.processed_root,
        original_video_key=args.original_video_key,
        use_original_video=args.use_original_video,
        enable_cache=args.enable_cache,
        extra_video_files={
            Path(f).stem: f for f in args.extra_videos
        },
        landmark_files={
            Path(f).stem: f for f in args.landmarks
        },
        self_connection_files={
            Path(f).stem: f for f in args.self_connection_files
        },
        other_connection_files={
            (Path(f).stem, Path(f).stem): f for f in args.other_connection_files
        },
    )
    dataset_to_zarr(fs50, args.export_path)

# Plugin info
info: Info = (FS50DatasetArgs, fs50_dataset_from_args)
"Plugin information for FluentSigners50 dataset integration."
