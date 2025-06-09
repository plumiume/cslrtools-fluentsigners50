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
