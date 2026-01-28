# Seminar Presentation Script: Weakly Supervised Anomaly Detection

**Time Estimate:** 3-5 Minutes
**Tone:** Professional, Confident, Technical

---

## 1. The Hook (Problem Statement)
"Good afternoon. We all know that video surveillance is everywhere—cameras are recording 24/7 in malls, airports, and streets. But here’s the problem: **Manual monitoring is impossible.** A single security guard cannot watch 50 screens at once.

We need *Automated Anomaly Detection*. But standard AI approaches fail here because they require **frame-level annotations**. Asking a human to draw a box around every single punch in a 10-hour video is too expensive and slow.

**Our Solution:** We built a **Weakly Supervised** system. 

---

## 2. Methodology: Multiple Instance Learning (The "Bag" Concept)
"We used a technique called **Multiple Instance Learning (MIL)**.
*   Think of a whole video as a **'Bag'**.
*   Think of the short clips inside it as **'Instances'** (or Segments).

**The Logic:**
*   An **Anomaly Video** contains *at least one* anomalous segment.
*   **The Problem:** Originally, we only had anomaly videos.
*   **The Fix:** We used our dataset's CSV metadata to identify specific **Normal Segments** (instances).
*   This allowed us to essentially "teach" the model what Normal looks like by making sure every batch included these verified normal instances (Label 0).

We train the model to rank the Anomaly Instances higher than the Normal Instances."

---

## 3. Architecture (Modernized)
"We modernized the classic 2018 approach (Sultani et al.) with 2025 tech:

1.  **Feature Extractor (ResNet50V2):**
    *   The original paper used C3D (old, heavy).
    *   We used **ResNet50V2 pretrained on ImageNet**. This gives us rich, semantic understanding of objects (arms, legs, weapons) right out of the box.

2.  **The MIL Scoring Head:**
    *   This is a custom Neural Network that takes those features and outputs a score from 0 to 1.
    *   **0 = Normal**, **1 = Anomaly**.

3.  **The Loss Function (The Secret Sauce):**
    *   We use a **Ranking Loss**. We tell the model: *'The highest score in the Fighting video MUST be higher than the highest score in the Normal video.'*
    *   We also added **Smoothness constraints** because real fights don't flicker on and off in 1 millisecond."

---

## 4. The Challenge: The "Cheating" Model
"During our experiments on the Hopper HPC, we hit a major wall.
*   **The Bug:** Our model achieved **100% Accuracy** immediately.
*   **The Reason:** We realized we were training *only* on anomaly videos. Without 'Normal' examples, the model just learned to scream 'ANOMALY!' for everything. It was cheating.
*   **The Fix:** We integrated the dataset's CSV metadata to force the model to see **True Negative** examples. Accuracy dropped to 50% (random guessing), and *then* it started actually learning. This proved our pipeline was valid."

---

## 5. Final Results
"After fixing that, and adding heavy regularization (Dropout and Noise), we achieved:
*   **Validation Accuracy:** **74.31%**
*   **Robustness:** The model is conservative. It gives scores of **~0.60** for fights (anomaly) and **~0.25** for walking (normal).

**Conclusion:** We successfully demonstrated that you don't need expensive frame-by-frame labeling. By treating videos as 'Bags of Segments', we can build scalable, automated security systems."

---

"Thank you. Questions?"
