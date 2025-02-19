# AI-music detection study
Code repository of our research paper on AI-generated music detection ["AI-Generated Music Detection and its Challenges"](https://arxiv.org/pdf/2501.10111) - D. Afchar, G. Meseguer Brocal, R. Hennequin (accepted for IEEE ICASSP 2025).

We create an AI-music detector by detecting the use of an artificial decoder (e.g., a neural decoder). For that, we auto-encode a dataset of music with several such auto-encoders to train on. This setting enables us to avoid detecting confounding artefacts. For instance, if a dataset of artificial music only contains pop music, you don't want to inadvertently train a pop music detector. Here, the task is to distinguish real music from its reconstructed counterpart. With the same musical content and compression setting, only the autoencoder artefacts remain. We also verify that merely training on autoencoder allows the model to detect music fully-generated from prompts (i.e., not auto-encoded).

Examples of audio reconstructions may be found in the `audio_examples` folder or on the demo page: [research.deezer.com/deepfake-detector/](https://research.deezer.com/deepfake-detector/).

The FMA dataset is available at [github.com/mdeff/fma](https://github.com/mdeff/fma).

More than a detector, we ponder the larger consequences of deploying a detector: robustness to manipulation, generalisation to different models, interpretability, ...

_Most of our experiment code is available for the review. We will make the trained weights open source for the publication._

⚠️ Following the recent press releases by Deezer on our [AI-music detection tool](https://newsroom-deezer.com/2025/01/deezer-deploys-cutting-edge-ai-detection-tool-for-music-streaming/), let us clarify something for interested readers: the tool available in this repository is **not** the tool we use in production for synthetic music detection. This is due to the delay between doing research and having a paper being published. Nevertheless, our new tool succeeds this present work, is elaborated by the same authors, and with the same concerns in mind, namely aiming for interpretability, almost perfect accuracy scores, and a focus on a possibility for recourse in case of false positives, generalisation to unknown scenarios and robustness to manipulation.
