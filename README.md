# Convolutional Learning on Directed Acyclic Graphs
Implementation of Convolutional GNNs for DAGs

This repository contains the code used in the paper "Convolutional Learning on Directed Acyclic Graphs" (pending acceptance) submitted at Asilomar 2024.

You can find the journal paper in arXiv: PENDING

The abstract of the paper reads as follows:

> We develop a novel convolutional architecture tailored for learning from data defined over directed acyclic graphs (DAGs). DAGs can be used to model causal relationships among variables, but their nilpotent adjacency matrices pose unique challenges towards developing DAG signal processing and machine learning tools. To address this limitation, we harness recent advances offering alternative definitions of causal shifts and convolutions for signals on DAGs. We develop a novel convolutional graph neural network that integrates learnable DAG filters to account for the partial ordering induced by the graph topology, thus providing valuable inductive bias to learn effective representations of DAG-supported data. We discuss the salient advantages and potential limitations of the proposed DAG convolutional network (DCN) and evaluate its performance on two learning tasks using synthetic data: network diffusion estimation and source identification. DCN compares favorably relative to several baselines, showcasing its promising potential.
