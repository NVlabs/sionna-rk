<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Sionna Research Kit: A GPU-Accelerated Research Platform for AI-RAN

> [!NOTE]
> The NVIDIA Jetson Thor as well as the NVIDIA DGX Spark will be supported soon. 


The NVIDIA Sionna&trade; Research Kit, powered by the [NVIDIA Jetson Platform](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/), is the pioneering solution that integrates in-line AI/ML accelerated computing with the adaptability of a software-defined Radio Access Network (RAN). Leveraging [OpenAirInterface](https://openairinterface.org), it guarantees [O-RAN](https://www.o-ran.org/)-compliant interfaces, providing extensive research opportunities—from 5G NR and [O-RAN](https://www.o-ran.org/) over real-world data acquisition to the deployment of cutting-edge [AI-RAN](https://ai-ran.org/) algorithms for 6G.

Created by the team behind [Sionna](https://github.com/NVlabs/sionna), the Sionna Research Kit features textbook-quality tutorials. In just an afternoon, you will connect commercial 5G equipment to a network using your own customizable receiver algorithms. Conducting AI-RAN experiments, whether cabled or over-the-air, has never been simpler and more cost-effective.

The official documentation can be found [here](https://nvlabs.github.io/sionna/rk).
You can build the documentation locally via ```make doc```. This may require to manually install `sudo apt install pandoc` and `pip install -r requirements_doc.txt`.

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
    version = {1.0.0}
   }
```

