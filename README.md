# Sperm-DFI-Classification
## 🧬 Detection and Classification of Sperm DNA Integrity
This project detects and classifies sperm cells in brightfield microscopy images based on their DNA Fragmentation Index (DFI). It integrates Meta AI's Segment Anything Model (SAM) for zero-shot sperm detection and a custom CNN for classifying DNA integrity into three categories:

Excellent (DFI < 15%)

Moderate (DFI 15–30%)

High Fragmentation (DFI > 30%)

We used preprocessing techniques like contrast enhancement and Gaussian filtering to improve image quality. Transfer learning was applied using a public dataset from McCallum et al. to address the small sample size. SAM showed robust performance in detection, while the CNN achieved competitive classification results.
