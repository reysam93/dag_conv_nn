# Convolutional Learning on Directed Acyclic Graphs

This repository contains the code related to the paper "Convolutional Learning on Directed Acyclic Graphs", by Samuel Rey, Hamed Ajorlou, and Gonzalo Mateos, presented at Asilomar 2024.

A preprint of the manuscript is available in arXiv: [Convolutional Learning on Directed Acyclic Graphs](https://arxiv.org/abs/2405.03056).


# Abstract

> We develop a novel convolutional architecture tailored for learning from data defined over directed acyclic graphs (DAGs). DAGs can be used to model causal relationships among variables, but their nilpotent adjacency matrices pose unique challenges towards developing DAG signal processing and machine learning tools. To address this limitation, we harness recent advances offering alternative definitions of causal shifts and convolutions for signals on DAGs. We develop a novel convolutional graph neural network that integrates learnable DAG filters to account for the partial ordering induced by the graph topology, thus providing valuable inductive bias to learn effective representations of DAG-supported data. We discuss the salient advantages and potential limitations of the proposed DAG convolutional network (DCN) and evaluate its performance on two learning tasks using synthetic data: network diffusion estimation and source identification. DCN compares favorably relative to several baselines, showcasing its promising potential.
