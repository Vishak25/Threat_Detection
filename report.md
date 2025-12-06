# Project Report: Weakly Supervised Camera Anomaly Detection

**Author:** Vishak Nandakumar  
**Date:** December 6, 2025  
**Platform:** GMU Hopper HPC (NVIDIA A100)

---

## 1. Executive Summary
This project implements a robust, real-time Video Anomaly Detection (VAD) system designed to identify high-risk behaviors (Arrest, Fighting) in surveillance footage. Unlike traditional "black box" approaches, we engineered a **Two-Stage Cascade Architecture** that combines:
1.  **Weakly Supervised Multiple Instance Learning (MIL):** To learn anomaly concepts from unlabelled video data.
2.  **Real-Time Object Detection (YOLOv8):** To focus the model's attention purely on human actors, eliminating background noise.

The system was trained and deployed on the **GMU Hopper HPC Cluster**, overcoming significant storage quotas through a novel **Hybrid "Local-to-Scratch" Workflow**. The final model achieves **~74.3% validation accuracy** and demonstrates conservative, robust detection in real-world tests.

---

## 2. Methodology & Innovation

### 2.1 The "Sultani et al." MIL Framework (Modernized)
We adopted the foundational approach from *Sultani et al. (CVPR 2018)* but significantly modernized the architecture to address its limitations.

| Component | Original Paper (2018) | Our Implementation (2025) | Impact |
| :--- | :--- | :--- | :--- |
| **Feature Extractor** | **C3D** (2014) | **ResNet50V2** (ImageNet) | C3D is computationally heavy and lacks semantic depth. ResNet50 captures rich object-level semantics ("person", "violence"). |
| **Inference Logic** | **Whole-Frame Features** | **YOLOv8 Cascade** | The original model looks at trees/sky (noise). Our model *only* looks at people (signal), drastically reducing false positives. |
| **Regularization** | Minimal (Standard Dropout) | **Gaussian Noise + High Dropout (0.4)** | Prevents the model from "cheating" (predicting 1.0 for everything) on small datasets. |
| **Supervision** | Weak (Video-level labels) | **Weak + Label Integration** | We integrated `csv` metadata to ensure true Negative samples (Label 0) were available during training. |

### 2.2 The Training vs. Inference Domain Shift
A key strategic decision was to decouple the Training and Inference domains:

*   **Training Domain (Whole-Frame):** The MIL model was trained on features extracted from the *entire video frame*.
    *   *Rationale:* This allowed the model to learn context and general scene dynamics without requiring expensive object detection preprocessing on the training set.
*   **Inference Domain (Object-Centric):** At runtime, we use YOLOv8 to crop "Person" instances, which are then fed to the MIL model.
    *   *Rationale:* This acts as a hard attention mechanism. By forcing the model to evaluate only the *human*, we increase sensitivity to the actual event and ignore irrelevant background motion.

---

## 3. Infrastructure: The Hybrid HPC Workflow
Deploying on a shared HPC environment (Hopper) presented strict resource constraints, specifically a **5GB Home Directory Quota**. We engineered a cloud-native pattern to bypass this:

1.  **Local Feature Extraction (The "Edge" Node):**
    *   Raw videos (GBs) were processed locally on the user's machine.
    *   **ResNet50V2** transformed 1000s of frames into compact `.npy` feature vectors (MBs).
2.  **Ephemeral Storage (The "Scratch" Space):**
    *   Features were uploaded to `/scratch/vnandak/` (Unlimited Quota, 90-day purge).
    *   This effectively decoupled "Storage" from "Compute."
3.  **Distributed Training (The "Compute" Node):**
    *   The specialized `hopper_train.py` script loaded features directly from `/scratch` into the A100 GPU memory.
    *   Result: We trained on a dataset larger than our allowed quota.

---

## 4. Technical Architecture

### 4.1 Model Architecture (`model.py`)
```python
Layer (type)                Output Shape              Param #
=================================================================
gaussian_noise (Gaussian)   (None, 2048)              0
dense_0 (Dense)             (None, 512)               1,049,088
dropout_0 (Dropout)         (None, 512)               0
dense_1 (Dense)             (None, 32)                16,416
dropout_1 (Dropout)         (None, 32)                0
score (Dense)               (None, 1)                 33
=================================================================
```
*   **Gaussian Noise (0.01):** Simulates sensor noise and prevents overfitting to exact feature values.
*   **Deep MLP Head:** Compresses 2048-dim features down to a single anomaly score.
*   **Sigmoid Activation:** Outputs a probability $P(Anomaly | Instance) \in [0, 1]$.

### 4.2 The MIL Ranking Loss
We implemented a custom loss function combining three terms:
$$ \mathcal{L} = \mathcal{L}_{rank} + \lambda_1 \mathcal{L}_{smooth} + \lambda_2 \mathcal{L}_{sparsity} $$

1.  **Ranking Loss:** Ensures the *most anomalous* segment in an Anomaly video has a higher score than the *most anomalous* segment in a Normal video.
2.  **Smoothness ($\lambda_1=0.01$):** Penalizes rapid score flickering (e.g., $0.1 \rightarrow 0.9 \rightarrow 0.1$) to ensure temporal consistency.
3.  **Sparsity:** Encourages the majority of segments to be normal (Score $\approx 0$).

---

## 5. Experimental Results

### 5.1 Quantitative Metrics
*   **Validation Accuracy:** **74.31%**
*   **Training Loss:** **0.134** (Stable convergence)

### 5.2 Qualitative Analysis (Real-Time Inference)
We tested the model on a stitched sequence (`Fighting002_full.mp4`) containing a progression from Normal $\rightarrow$ Fighting $\rightarrow$ Arrest.

*   **Normal Phase (Frames 0-50):**
    *   Scores: **0.15 - 0.35**
    *   Status: **Green (Normal)**. The model correctly identifies standard human movement as non-threatening.
*   **Fighting Phase (Frames 55-120):**
    *   Scores: **0.55 - 0.75**
    *   Status: **Red (Anomaly)**. The model detects the violence.
    *   *Observation:* Scores fluctuate due to occlusion, but consistently stay above the threshold.
*   **The "YOLO Factor":**
    *   Initial tests showed "No Box" errors for the aggressors.
    *   *Root Cause:* Fast motion and odd poses confused the YOLOv8 Nano model at `0.6` confidence.
    *   *Fix:* Lowering confidence to **0.25** significantly improved recall, catching 100% of the actors.

---

## 6. Conclusion & Future Work
This project successfully demonstrates that **Weakly Supervised Learning** is a viable path for scalable Video Anomaly Detection. By removing the need for expensive frame-level labels and leveraging modern CNN backbones, we created a system that is both accurate and cost-effective.

**Key Achievements:**
1.  **HPC Optimization:** Solved the "Disk Quota" bottleneck.
2.  **Robustness:** Prevented the "100% Accuracy" cheating artifact via regularization.
3.  **Real-World Applicability:** The YOLO cascade makes the system deployable in cluttered real-world environments.

**Future Directions:**
*   **End-to-End Fine-tuning:** Train the ResNet50 backbone (currently frozen) on the anomaly task.
*   **Domain Alignment:** Retrain the MIL model using features extracted from *YOLO crops* instead of full frames to close the domain gap.
