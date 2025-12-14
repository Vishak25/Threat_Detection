# Project Report: Weakly Supervised Camera Anomaly Detection

**Author:** Vishak Nandakumar  
**Date:** December 6, 2025  
**Platform:** GMU Hopper HPC (NVIDIA A100)

---

## 1. Executive Summary
This project implements a robust Video Anomaly Detection (VAD) system designed to identify high-risk behaviors (Arrest, Fighting) in surveillance footage using **Weakly Supervised Multiple Instance Learning (MIL)**. 

The system was trained and deployed on the **GMU Hopper HPC Cluster**. We successfully overcame significant infrastructure constraints—specifically strict storage quotas—by engineering a **Hybrid "Local-to-Scratch" Workflow**. The final model, trained on the DCSASS dataset, achieves **~74.3% validation accuracy** and demonstrates a capability to reliably discriminate between normal and anomalous video segments without requiring expensive frame-level supervision.

---

## 2. Methodology

### 2.1 The "Sultani et al." MIL Framework (Modernized)
We adopted the foundational approach from *Sultani et al. (CVPR 2018)*, where videos are treated as "bags" of instances (segments). A video labeled "Anomaly" contains at least one anomalous segment, while a "Normal" video contains only normal segments. We modernized the original architecture to improve performance and stability:

| Component | Original Paper (2018) | Our Implementation (2025) | Impact |
| :--- | :--- | :--- | :--- |
| **Feature Extractor** | **C3D** (2014) | **ResNet50V2** (ImageNet) | C3D is computationally heavy. ResNet50 captures deeper spatial semantics, allowing the model to recognize violence cues (posture, objects) more effectively. |
| **Regularization** | Minimal | **Gaussian Noise + High Dropout (0.4)** | Prevents the model from "cheating" (predicting 1.0 for everything) on small datasets. |
| **Supervision** | Weak (Video-level labels) | **Weak + Label Integration** | We integrated `csv` metadata to ensure true Negative samples (Label 0) were properly utilized during training, resolving a critical class imbalance. |

### 2.2 The MIL Ranking Loss
We implemented a custom loss function designed to widen the gap between normal and anomalous scores:

$$ \mathcal{L} = \mathcal{L}_{rank} + \lambda_1 \mathcal{L}_{smooth} + \lambda_2 \mathcal{L}_{sparsity} $$

1.  **Ranking Loss:** Enforces that the *maximum* anomaly score in an Anomaly video is significantly higher than the maximum score in a Normal video.
2.  **Smoothness ($\lambda_1=0.01$):** Penalizes rapid score flickering to ensure temporal consistency (real-world anomalies are continuous, not single-frame blips).
3.  **Sparsity:** Encourages the majority of segments in an anomaly video to be normal, reflecting the reality that anomalies are rare events.

---

## 3. Experimental Flow & Infrastructure

### 3.1 Infrastructure Challenge: The "Disk Quota" Barrier
*   **The Problem:** The DCSASS dataset exceeded the strict 5GB Home directory quota on the Hopper cluster, preventing standard data uploading.
*   **The Solution (Hybrid Workflow):** 
    1.  **Local Feature Extraction:** We processed raw videos locally, using ResNet50V2 to extract only the high-level features (`.npy` files), reducing data size by ~95%.
    2.  **Scratch Space Utilization:** We uploaded these lightweight features to the `/scratch` directory (Unlimited Quota) on Hopper.
    3.  **A100 Training:** The training script was optimized to load data directly from ephemeral scratch storage into GPU memory.

### 3.2 Experiment 1: The "One-Class" Artifact
*   **Observation:** Initial training runs achieved **100% Accuracy** almost immediately.
*   **Analysis:** The model had collapsed into a trivial solution, predicting "1.0" (Anomaly) for every input. This occurred because the training set initially lacked explicit "Normal" videos, breaking the ranking loss logic.
*   **Refinement:** We integrated the full dataset metadata (`Labels/*.csv`) to correctly identify and include purely normal segments. This forced the model to learn a discriminative boundary, dropping accuracy to a realistic ~50% initially before learning commenced.

### 3.3 Experiment 2: Combating Overfitting
*   **Observation:** With a limited subset of videos, the model began memorizing specific feature vectors.
*   **Refinement:** We introduced aggressive regularization:
    *   **Gaussian Noise Injection (std=0.01):** Added to input features to simulate sensor noise and data diversity.
    *   **Increased Dropout (0.4):** Applied to the MIL scoring head.
    *   **Result:** The model became more conservative. Instead of confidently predicting 0.99 for every fight, it predicted robust scores in the 0.60–0.70 range, indicating true generalization rather than memorization.

---

## 4. Final Results

### 4.1 Quantitative Metrics
*   **Validation Accuracy:** **74.31%**
*   **Training Loss:** **0.134**
*   **Status:** Stable convergence with no signs of mode collapse.

### 4.2 Qualitative Analysis
We analyzed the model's predictions on test sequences containing known transitions from normal behavior to fighting and arrest.

*   **Normal Segments:** The model consistently output low anomaly scores (**0.15 - 0.35**), correctly identifying standard human movement as non-threatening.
*   **Anomalous Segments (Fighting/Arrest):** The model successfully detected violent events, producing elevated scores (**0.55 - 0.75**). 
*   **Robustness:** The implementation of the Smoothness constraint effectively filtered out single-frame noise, resulting in stable, continuous detection blocks during the anomaly phase.

---

## 5. Conclusion
This project successfully establishes a scientifically valid Video Anomaly Detection pipeline on the Hopper HPC environment. By modernizing the MIL framework with a ResNet50 backbone and implementing a robust regularization strategy, we achieved a reliable detector that operates effectively under weak supervision. The Hybrid Workflow developed here serves as a scalable template for processing large-scale video datasets on resource-constrained HPC clusters.
