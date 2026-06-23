# SINFONY: Semantic INFOrmation traNsmission and recoverY


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8006567.svg)](https://doi.org/10.5281/zenodo.8006567) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20813869.svg)](https://doi.org/10.5281/zenodo.20813869)

Source code from the PhD thesis [1] and the scientific research articles [2, 3] about the semantic communication approach SINFONY:

1. Edgar Beck, Advancing Semantic and Digital Communications through Machine Learning, ser. Dissertations from the Department of Communications Engineering, University of Bremen, A. Dekorsy, Ed. Düren: Shaker Verlag, Dec. 2025, vol. 15. https://doi.org/10.26092/elib/4791

2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347
(First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022)

3. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1 pp. 7, Stockholm, Sweden, May 2024. https://doi.org/10.1109/ICMLCN59089.2024.10625190

SINFONY and the Generalized Context Model (GCM) implementation are part of the End-to-End Sensing-Decision Framework in:

4. E. Beck, H.-Y. Lin, P. Rückert, Y. Bao, B. von Helversen, S. Fehrler, K. Tracht, and A. Dekorsy, “Integrating Semantic Communication and Human Decision-Making into an End-to-End Sensing-Decision Framework,” IEEE Open Journal of the Communications Society, vol. 7, pp. 748-768, Jan 2026. https://doi.org/10.1109/OJCOMS.2026.3652845

## Related Repositories

[`semantic-floating-point`](https://github.com/ant-uni-bremen/semantic-floating-point): The source code of the proposed semantics-aware, i.e., seismic exploration data-aware, receiver in [5] has been moved and can be found in this repository.

5. Edgar Beck, Ban-Sok Shin, Shengdi Wang, Thomas Wiedemann, Dmitriy Shutin, and Armin Dekorsy, “Swarm Exploration and Communications: A First Step towards Mutually-Aware Integration by Probabilistic Learning,” MDPI Electronics, vol. 12, no. 8, p. 1908, Apr. 2023. https://doi.org/10.3390/electronics12081908

# Requirements & Usage

- The code base has been updated to `Keras 3` including all model saves and is now **backend-agnostic**, meaning that it can run with both `tensorflow` and `pytorch`. 

- You can test successful conversion and reproduction running the scripts `sinfony_test.py` and `gcm_test_models_conversion.py` that are able to reproduce all the results except for a few that require special settings.

- For the script `sinfony_classic.py`, further sionna (>=1.0.0) is required.

Run the script as `python3 sinfony.py "semantic_config"`, `python3 sinfony_classic.py "classic/config_classic"`, `python3 sinfony_classic_features_autoencoder.py "classic/config_classic_features_autoencoder"`, or `python3 gcm.py` to reproduce the results of the articles. To do so, set the parameters in the configuration files to the values in the articles. Exemplary configurations are given in the folder `settings` and its subfolders named according to the used datasets. For GCM simulations, set the parameters directly in the script file.

## Dependencies
The dependencies are provided in the conda environment file `conda_env_keras3.yaml` for the TensorFlow backend and `conda_env_keras3_pytorch.yaml` for the PyTorch backend. Dependencies from older SINFONY versions that were used throughout the years can be found in the git history and on Zenodo.

### SINFONY model weight and simulation result files

This repository uses **Git Large File Storage (LFS)** to manage large SINFONY model weight and simulation result files.

After cloning the repository, the files tracked by LFS may appear as small placeholder files. To download the actual file contents, Git LFS must be installed and initialized.

#### Setup

Install Git LFS:
https://git-lfs.com

Then run:

```bash
git lfs install
git lfs pull
```

This will fetch all large files tracked by LFS into your local repository.

If Git LFS is not installed, only placeholder references to the files will be available, and the actual content will not be accessible.

### Alternative: SINFONY Dataset

Alternatively, you can find the SINFONY model weights and simulation results (v3.0.0) also in a Zenodo dataset: https://doi.org/10.5281/zenodo.20813869

After downloading the zip-File, extract the contents of `models_selected/` to the folder `models/`.

# Acknowledgements

This work was partly funded by the Federal State of Bremen and the University of Bremen as part of the Humans on Mars Initiative, by the German Ministry of Education and Research (BMBF) under grant 16KISK016 (Open6GHub), and by the German Research Foundation (DFG) under grant 500260669 (SCIL).

# License and Referencing

This program is licensed under the GPLv3 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

# Abstract of the articles

2. Motivated by the recent success of Machine Learning (ML) tools in wireless communications, the idea of semantic communication by Weaver from 1949 has gained attention. It breaks with Shannon's classic design paradigm by aiming to transmit the meaning of a message, i.e., semantics, rather than its exact version and thus allows for savings in information rate. In this work, we extend the fundamental approach from Basu et al. for modeling semantics to the complete communications Markov chain. Thus, we model semantics by means of hidden random variables and define the semantic communication task as the data-reduced and reliable transmission of messages over a communication channel such that semantics is best preserved. We cast this task as an end-to-end Information Bottleneck problem, allowing for compression while preserving relevant information most. As a solution approach, we propose the ML-based semantic communication system SINFONY and use it for a distributed multipoint scenario: SINFONY communicates the meaning behind multiple messages that are observed at different senders to a single receiver for semantic recovery. We analyze SINFONY by processing images as message examples. Numerical results reveal a tremendous rate-normalized SNR shift up to 20 dB compared to classically designed communication systems.

3. Following the recent success of Machine Learning tools in wireless communications, the idea of semantic communication by Weaver from 1949 has gained attention. It breaks with Shannon's classic design paradigm by aiming to transmit the meaning, i.e., semantics, of a message instead of its exact version, allowing for information rate savings. In this work, we apply the Stochastic Policy Gradient (SPG) to design a semantic communication system by reinforcement learning, separating transmitter and receiver, and not requiring a known or differentiable channel model -- a crucial step towards deployment in practice. Further, we derive the use of SPG for both classic and semantic communication from the maximization of the mutual information between received and target variables. Numerical results show that our approach achieves comparable performance to a model-aware approach based on the reparametrization trick, albeit with a decreased convergence rate.

4. As early as 1949, Weaver defined communication in a very broad sense to include all procedures by which one mind or technical system can influence another, thus establishing the idea of semantic communication. With the recent success of machine learning in expert assistance systems where sensed information is wirelessly provided to a human to assist task execution, the need to design effective and efficient communications has become increasingly apparent. In particular, semantic communication aims to convey the meaning behind the sensed information relevant for Human Decision-Making (HDM). Regarding the interplay between semantic communication and HDM, many questions remain, such as how to model the entire end-to-end sensing-decision-making process, how to design semantic communication for the HDM and which information should be provided for HDM. To address these questions, we propose to integrate semantic communication and HDM into one probabilistic end-to-end sensing-decision framework that bridges communications and psychology. In our interdisciplinary framework, we model the human through a HDM process, allowing us to explore how feature extraction from semantic communication can best support HDM both in theory and in simulations. In this sense, our study reveals the fundamental design trade-off between maximizing the relevant semantic information and matching the cognitive capabilities of the HDM model. Our initial analysis shows how semantic communication can balance the level of detail with human cognitive capabilities while demanding less bandwidth, power, and latency.


