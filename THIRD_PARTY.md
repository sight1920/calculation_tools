# Third-Party Materials

This source distribution contains the project's Python metric wrappers, evaluation entry point, setup scripts and documentation. Third-party source archives, compiled libraries, model weights and example media are obtained separately by the setup process. Their download URLs, sizes and SHA256 values are recorded in `assets.json`.

| Component | Source/version | Distribution note |
| --- | --- | --- |
| VMAF | Netflix v3.2.1 | Upstream BSD-2-Clause-Patent license is copied by the installer into `third_party/vmaf-3.2.1/LICENSE`. |
| FDIM | avs-cvqa commit `e01944a363089a636ed3ce092b1d20cf475d44f2` | This snapshot contains a checkpoint and VMAF 3.0.0 binaries. No explicit license declaration was found in the inspected snapshot's license files, README or package metadata. Confirm the applicable permission with the upstream owner before redistributing those materials. |
| LPIPS/DISTS implementation | PyIQA 0.1.15.post2 | Installed as a Python dependency; retain its upstream notices and applicable model terms. |
| AlexNet/VGG16/LPIPS/DISTS weights | Official PyTorch and PyIQA downloads | Downloaded separately; not relicensed by this project. |
| Example YUV pair | Netflix/vmaf_resource commit `c0ab6adbd7e41bb354f14686ed08500530622bc3` | Downloaded separately; source ownership and terms remain with the upstream provider. |
| Color conversion convention | Microsoft DCVC commit `54e88645e1b4edac0694228ab2f44ed8818e345b` | The README identifies the source of the agreed conversion convention. |

The project owner's license and copyright attribution have not been selected in this packaging task. A public GitHub release should state that selection explicitly. Downloading an upstream component does not grant permission to relicense it.
