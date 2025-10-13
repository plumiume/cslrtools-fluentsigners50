# Copyright 2025 plumiume.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from clipar import namespace

@namespace
class Args:
    root: str
    "Root directory of the FluentSigners50 dataset."
    landmarks: str
    "Path to the landmarks file."
    dst: str
    "Destination path to save the dataset."
    use_translation: bool = False
    "Whether to use translation in the dataset."

def main_impl(ns: Args.T):

    from halo import Halo # pyright: ignore[reportMissingTypeStubs]

    with Halo(text='Waiting for Package import...', spinner='dots'):
        from .pytorch import FluentSigners50 
    print('✅ Waiting for Package import finished.')

    dataset = FluentSigners50(
        root=ns.root,
        landmarks=ns.landmarks,
        use_translation=ns.use_translation
    )

    dataset.save(ns.dst)

def main():

    ns = Args.parse_args()

    main_impl(ns)

if __name__ == '__main__':
    main()