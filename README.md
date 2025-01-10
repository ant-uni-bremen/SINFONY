# SINFONY: Semantic INFOrmation traNsmission and recoverY

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8006567.svg)](https://doi.org/10.5281/zenodo.8006567)

Source code from scientific research articles [1, 2] about the semantic communication approach SINFONY:

1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Semantic Information Recovery in Wireless Networks,” MDPI Sensors, vol. 23, no. 14, p. 6347, 2023. https://doi.org/10.3390/s23146347
(First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022)

2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,” in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN 2024), vol. 1 pp. 7, Stockholm, Sweden, May 2024. https://doi.org/10.1109/ICMLCN59089.2024.10625190

Further, the source code of the proposed semantics-aware, i.e., seismic exploration data-aware, receiver in [3] can be found here:

3. Edgar Beck, Ban-Sok Shin, Shengdi Wang, Thomas Wiedemann, Dmitriy Shutin, and Armin Dekorsy, “Swarm Exploration and Communications: A First Step towards Mutually-Aware Integration by Probabilistic Learning,” MDPI Electronics, vol. 12, no. 8, p. 1908, Apr. 2023. https://doi.org/10.3390/electronics12081908

SINFONY is also part of the End-to-End Sensing-Decision Framework in:

4. E. Beck, H.-Y. Lin, P. Rückert, Y. Bao, B. von Helversen, S. Fehrler, K. Tracht, and A. Dekorsy, “Integrating Semantic Communication and Human Decision-Making into an End-to-End Sensing-Decision Framework”, arXiv preprint: 2412.05103, Dec. 2024. https://doi.org/10.48550/arXiv.2412.05103

# Requirements & Usage

This code was tested with TensorFlow 2.6 and should also run with version 2.10-2.15. For the script `sinfony_classic.py`, further sionna (>=0.9.0) is required.

Run the script as `python3 sinfony.py "semantic_config.yaml"`, `python3 sinfony_classic.py "classic/config_classic.yaml"`, `python3 sinfony_classic_features_autoencoder.py "classic/config_classic_features_autoencoder.yaml"`, or `python3 semantic_floating_point.py` to reproduce the results of the articles. To do so, set the parameters in the configuration files to the values in the articles. Exemplary configurations are given in the folder `settings` and its subfolders named according to the used datasets.

# Acknowledgements

This work was partly funded by the Federal State of Bremen and the University of Bremen as part of the Humans on Mars Initiative, by the German Ministry of Education and Research (BMBF) under grant 16KISK016 (Open6GHub), and by the German Research Foundation (DFG) under grant 500260669 (SCIL).

# License and Referencing

This program is licensed under the GPLv3 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

# Abstract of the articles

1. Motivated by the recent success of Machine Learning (ML) tools in wireless communications, the idea of semantic communication by Weaver from 1949 has gained attention. It breaks with Shannon's classic design paradigm by aiming to transmit the meaning of a message, i.e., semantics, rather than its exact version and thus allows for savings in information rate. In this work, we extend the fundamental approach from Basu et al. for modeling semantics to the complete communications Markov chain. Thus, we model semantics by means of hidden random variables and define the semantic communication task as the data-reduced and reliable transmission of messages over a communication channel such that semantics is best preserved. We cast this task as an end-to-end Information Bottleneck problem, allowing for compression while preserving relevant information most. As a solution approach, we propose the ML-based semantic communication system SINFONY and use it for a distributed multipoint scenario: SINFONY communicates the meaning behind multiple messages that are observed at different senders to a single receiver for semantic recovery. We analyze SINFONY by processing images as message examples. Numerical results reveal a tremendous rate-normalized SNR shift up to 20 dB compared to classically designed communication systems.

2. Following the recent success of Machine Learning tools in wireless communications, the idea of semantic communication by Weaver from 1949 has gained attention. It breaks with Shannon's classic design paradigm by aiming to transmit the meaning, i.e., semantics, of a message instead of its exact version, allowing for information rate savings. In this work, we apply the Stochastic Policy Gradient (SPG) to design a semantic communication system by reinforcement learning, separating transmitter and receiver, and not requiring a known or differentiable channel model -- a crucial step towards deployment in practice. Further, we derive the use of SPG for both classic and semantic communication from the maximization of the mutual information between received and target variables. Numerical results show that our approach achieves comparable performance to a model-aware approach based on the reparametrization trick, albeit with a decreased convergence rate.

3. Swarm exploration by multi-agent systems relies on stable inter-agent communication. However, so far both exploration and communication have been mainly considered separately despite their strong inter-dependency in such systems. In this paper, we present the first steps towards a framework that unifies both of these realms by a “tight” integration. We propose to make exploration “communication-aware” and communication “exploration-aware” by using tools of probabilistic learning and semantic communication, thus enabling the coordination of knowledge and action in multi-agent systems. We anticipate that by a “tight” integration of the communication chain, the exploration strategy will balance the inference objective of the swarm with exploration-tailored, i.e., semantic, inter-agent communication. Thus, by such a semantic communication design, communication efficiency in terms of latency, required data rate, energy, and complexity may be improved. With this in mind, the research proposed in this work addresses challenges in the development of future distributed sensing and data processing platforms—sensor networks or mobile robotic swarms consisting of multiple agents—that can collect, communicate, and process spatially distributed sensor data.

4. As early as 1949, Weaver defined communication in a very broad sense to include all procedures by which one mind or technical system can influence another, thus establishing the idea of semantic communication. With the recent success of machine learning in expert assistance systems where sensed information is wirelessly provided to a human to assist task execution, the need to design effective and efficient communications has become increasingly apparent. In particular, semantic communication aims to convey the meaning behind the sensed information relevant for Human Decision-Making (HDM). Regarding the interplay between semantic communication and HDM, many questions remain, such as how to model the entire end-to-end sensing-decision-making process, how to design semantic communication for the HDM and which information should be provided to the HDM. To address these questions, we propose to integrate semantic communication and HDM into one probabilistic end-to-end sensing-decision framework that bridges communications and psychology. In our interdisciplinary framework, we model the human through a HDM process, allowing us to explore how feature extraction from semantic communication can best support human decision-making. In this sense, our study provides new insights for the design/interaction of semantic communication with models of HDM. Our initial analysis shows how semantic communication can balance the level of detail with human cognitive capabilities while demanding less bandwidth, power, and latency.
