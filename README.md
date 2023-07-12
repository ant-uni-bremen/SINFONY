# SINFONY: Semantic INFOrmation traNsmission and recoverY

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8006567.svg)](https://doi.org/10.5281/zenodo.8006567)

Source code from scientific research articles [1, 2] about the semantic communication approach SINFONY:

1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Semantic Information Recovery in Wireless Networks," https://doi.org/10.48550/arXiv.2204.13366
(First draft version: E. Beck, C. Bockelmann, and A. Dekorsy, “Semantic communication: An information bottleneck view,” arXiv:2204.13366, Apr. 2022)

2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient," https://doi.org/10.48550/arXiv.2305.03571

Further, the source code of the proposed semantics-aware, i.e., seismic exploration data-aware, receiver in [3] can be found here:

3. E. Beck, B.-S. Shin, S. Wang, T. Wiedemann, D. Shutin, and A. Dekorsy, “Swarm Exploration and Communications: A First Step towards Mutually-Aware Integration by Probabilistic Learning,” Electronics, vol. 12, no. 8, p. 1908, Apr. 2023

## Abstract of the articles

1. Motivated by the recent success of Machine Learning (ML) tools in wireless communications, the idea of semantic communication by Weaver from 1949 has gained attention. It breaks with Shannon's classic design paradigm by aiming to transmit the meaning of a message, i.e., semantics, rather than its exact version and thus allows for savings in information rate. In this work, we extend the fundamental approach from Basu et al. for modeling semantics to the complete communications Markov chain. Thus, we model semantics by means of hidden random variables and define the semantic communication task as the data-reduced and reliable transmission of messages over a communication channel such that semantics is best preserved. We cast this task as an end-to-end Information Bottleneck problem, allowing for compression while preserving relevant information most. As a solution approach, we propose the ML-based semantic communication system SINFONY and use it for a distributed multipoint scenario: SINFONY communicates the meaning behind multiple messages that are observed at different senders to a single receiver for semantic recovery. We analyze SINFONY by processing images as message examples. Numerical results reveal a tremendous rate-normalized SNR shift up to 20 dB compared to classically designed communication systems.

2. Motivated by the recent success of Machine Learning tools in wireless communications, the idea of semantic communication by Weaver from 1949 has gained attention. It breaks with Shannon's classic design paradigm by aiming to transmit the meaning, i.e., semantics, of a message instead of its exact version, allowing for information rate savings. In this work, we apply the Stochastic Policy Gradient (SPG) to design a semantic communication system by reinforcement learning, not requiring a known or differentiable channel model - a crucial step towards deployment in practice. Further, we motivate the use of SPG for both classic and semantic communication from the maximization of the mutual information between received and target variables. Numerical results show that our approach achieves comparable performance to a model-aware approach based on the reparametrization trick, albeit with a decreased convergence rate.

3. Swarm exploration by multi-agent systems relies on stable inter-agent communication. However, so far both exploration and communication have been mainly considered separately despite their strong inter-dependency in such systems. In this paper, we present the first steps towards a framework that unifies both of these realms by a “tight” integration. We propose to make exploration “communication-aware” and communication “exploration-aware” by using tools of probabilistic learning and semantic communication, thus enabling the coordination of knowledge and action in multi-agent systems. We anticipate that by a “tight” integration of the communication chain, the exploration strategy will balance the inference objective of the swarm with exploration-tailored, i.e., semantic, inter-agent communication. Thus, by such a semantic communication design, communication efficiency in terms of latency, required data rate, energy, and complexity may be improved. With this in mind, the research proposed in this work addresses challenges in the development of future distributed sensing and data processing platforms—sensor networks or mobile robotic swarms consisting of multiple agents—that can collect, communicate, and process spatially distributed sensor data.

# Requirements & Usage

This code was tested with TensorFlow 2.6 and should also run with version 2.10. For the script "SINFONY_classic", further sionna (>=0.9.0) is required.

Run the script as python3 SINFONY.py, python3 SINFONY_classic.py or python3 SemFloat.py to reproduce the results of the articles. To do so, set the parameters to the values in the articles.

# Acknowledgements

This work was partly funded by the Federal State of Bremen and the University of Bremen as part of the Human on Mars Initiative, and by the German Ministry of Education and Research (BMBF) under grant 16KISK016 (Open6GHub).

# License and Referencing

This program is licensed under the GPLv3 license. If you in any way use this code for research that results in publications, please cite our original article listed above.
