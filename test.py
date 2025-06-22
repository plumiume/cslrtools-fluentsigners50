from argparse_class_namespace import namespace

@namespace
class Args:
    ptfile: str

if __name__ == "__main__":

    ns = Args.parse_args()

    import torch
    from cslrtools_fluentsigners50.pytorch import FluentSigners50

    fs50: FluentSigners50 = torch.load(ns.ptfile, weights_only=False)
    # python -i test.py <ptfile>