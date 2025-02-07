# AI-music detection study
Code repository of our research paper on AI-generated music detection - D. Afchar, G. Meseguer Brocal, R. Hennequin (Accepted for ICASSP 2025).

Examples of audio reconstructions may be found in the `audio_examples` folder or on the demo page: [research.deezer.com/deepfake-detector/](https://research.deezer.com/deepfake-detector/).

The FMA dataset is available at [github.com/mdeff/fma](https://github.com/mdeff/fma).

_Most of our experiment code is available for the review. We will make the trained weights open source for the publication._

⚠️ Following the recent press releases by Deezer on our [AI detection tool](https://newsroom-deezer.com/2025/01/deezer-deploys-cutting-edge-ai-detection-tool-for-music-streaming/), let us clarify something for interested readers: the tool available in this repository is **not** the tool we use in production for deepfake detection. This is due to the delay between doing research and having a paper being published. Nevertheless, our new tool succeeds to this present work, elaborated by the same authors, and with the same concerns in mind, namely aiming for interpretability, almost perfect accuracy scores, and a focus on a possibility for recourse in case of false positives, generalisation to unknown scenarios and robustness to manipulation.
