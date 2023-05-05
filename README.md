# SINFONY: Semantic INFOrmation traNsmission and recoverY

[The code will be uploaded within the next weeks.]

Source code behind the scientific research articles about the semantic communication approach SINFONY:

Edgar Beck, Carsten Bockelmann and Armin Dekorsy, "Semantic Information Recovery in Wireless Networks," 10.48550/arXiv.2204.13366

Edgar Beck, Carsten Bockelmann and Armin Dekorsy, "Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient,"

## Abstract of the article

Motivated by the recent success of Machine Learning (ML) tools in wireless communications, the idea of semantic communication by Weaver from 1949 has received considerable attention. It breaks with the classic design paradigm of Shannon by aiming to transmit the meaning of a message, i.e., semantics, rather than its exact copy and thus allows for savings in information rate. In this work, we extend the fundamental approach from Basu et al. for modeling semantics to the complete communications Markov chain. Thus, we model semantics by means of hidden random variables and define the semantic communication task as the data-reduced and reliable transmission of messages over a communication channel such that semantics is best preserved. We cast this task as an end-toend Information Bottleneck problem allowing for compression while preserving relevant information at most. As a solution approach, we propose the ML-based semantic communication system SINFONY and use it for a distributed multipoint scenario: SINFONY communicates the meaning behind multiple messages that are observed at different senders to a single receiver for semantic recovery. We analyze SINFONY by processing images as message examples. Numerical results reveal a tremendous rate-normalized SNR shift up to 20 dB compared to classically designed communication systems.


# Requirements & Usage

This code was tested with tensorflow 2.6 and should also run with version 2.10. 

Run the script as python3 SINFONY.py to reproduce the results of the article. To do so, set the parameters to the values in the article.

# Acknowledgements

This work was partly funded by the Federal State of Bremen and the University of Bremen as part of the Human on Mars Initiative.

# License and Referencing

This program is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.
