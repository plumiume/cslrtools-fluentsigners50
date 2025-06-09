from argparse_class_namespace import namespace
from . import FluentSigners50, _PathLike

def main():

    @namespace
    class Args:
        root: _PathLike
        landmarks: str
        dst: _PathLike # .pkl
        use_translation: bool = False

    ns = Args.parse_args()

    dataset = FluentSigners50(
        root=ns.root,
        landmarks=ns.landmarks,
        use_translation=ns.use_translation
    )

    dataset.save(ns.dst)
