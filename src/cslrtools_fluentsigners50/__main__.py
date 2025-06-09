def main():

    from halo import Halo

    with Halo(text='Waiting for Package import...', spinner='dots'):
        from . import FluentSigners50
        from argparse_class_namespace import namespace
    print('✅ Waiting for Package import finished.')

    @namespace
    class Args:
        root: str
        landmarks: str
        dst: str
        use_translation: bool = False

    ns = Args.parse_args()

    dataset = FluentSigners50(
        root=ns.root,
        landmarks=ns.landmarks,
        use_translation=ns.use_translation
    )

    dataset.save(ns.dst)
