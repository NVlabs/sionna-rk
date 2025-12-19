<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Sionna Research Kit: A GPU-Accelerated Research Platform for AI-native RAN

The [Sionna Research Kit](https://github.com/NVlabs/sionna-rk) is an accelerated, open platform for wireless research and development. Powered by the [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) and built on [OpenAirInterface](https://openairinterface.org), it provides a software-defined 5G RAN and core network for end-to-end experimentation running in real-time.

Created by the team behind [Sionna](https://github.com/NVlabs/sionna), it features textbook-quality tutorials and [O-RAN](https://www.o-ran.org/)-compliant interfaces. In just one afternoon, you will connect commercial 5G equipment to a network using your own customizable transceiver algorithms. Conducting [AI-RAN](https://ai-ran.org/) experiments, whether simulated, cabled, or over-the-air, has never been more accessible.

The official documentation can be found [here](https://nvlabs.github.io/sionna/rk).
You can build the documentation locally via ```make doc```. This may require manually installing `sudo apt install pandoc` and `pip install -r requirements_doc.txt`.

See the [Quickstart Guide](https://nvlabs.github.io/sionna/rk/quickstart.html) to get started.

## License and Citation

The NVIDIA Sionna Research Kit is licensed under the Apache 2.0 license, as found in the LICENSE file.

In connection with your use of this software, you may receive links to third party technology, and your use of third-party technology may be subject to third-party terms, privacy statements or practices. NVIDIA is not responsible for the terms, privacy statements or practices of third parties. You acknowledge and agree that it is your sole responsibility to obtain any additional third-party licenses required to make, have made, use, have used, sell, import, and offer for sale products or services that include or incorporate any third-party technology. NVIDIA does not grant to you under the project license any necessary patent or other rights, including standard essential patent rights, with respect to any such third-party technology.

If you use this software, please cite it as:
```bibtex
   @software{sionna-rk,
    title = {Sionna Research Kit},
    author = {Cammerer, Sebastian, and Marcus, Guillermo and Zirr, Tobias and Hoydis, Jakob and {Ait Aoudia}, Fayçal and Wiesmayr, Reinhard and Maggi, Lorenzo and Nimier-David, Merlin and Keller, Alexander},
    note = {https://nvlabs.github.io/sionna/rk/index.html},
    year = {2025},
    version = {1.1.0}
   }
```

