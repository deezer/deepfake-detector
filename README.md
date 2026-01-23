# AI-music detection study
Code repository of our research paper on AI-generated music detection ["AI-Generated Music Detection and its Challenges"](https://arxiv.org/pdf/2501.10111) - D. Afchar, G. Meseguer Brocal, R. Hennequin (accepted for IEEE ICASSP 2025).

We create an AI-music detector by detecting the use of an artificial decoder (e.g., a neural decoder). For that, we auto-encode a dataset of music with several such auto-encoders to train on. This setting enables us to avoid detecting confounding artefacts. For instance, if a dataset of artificial music only contains pop music, you don't want to inadvertently train a pop music detector. Here, the task is to distinguish real music from its reconstructed counterpart. With the same musical content and compression setting, only the autoencoder artefacts remain. We also verify that merely training on autoencoder allows the model to detect music fully-generated from prompts (i.e., not auto-encoded).

Examples of audio reconstructions may be found in the `audio_examples` folder or on the demo page: [research.deezer.com/deepfake-detector/](https://research.deezer.com/deepfake-detector/).

The FMA dataset is available at [github.com/mdeff/fma](https://github.com/mdeff/fma).

More than a detector, we ponder the larger consequences of deploying a detector: robustness to manipulation, generalisation to different models, interpretability, ...

⚠️ Following the recent press releases by Deezer on our [AI-music detection tool](https://newsroom-deezer.com/2025/01/deezer-deploys-cutting-edge-ai-detection-tool-for-music-streaming/), let us clarify something for interested readers: the tool available in this repository is **not** the tool we use in production for synthetic music detection. This is due to the delay between doing research and having a paper being published. Nevertheless, our new tool succeeds this present work, is elaborated by the same authors, and with the same concerns in mind, namely aiming for interpretability, almost perfect accuracy scores, and a focus on a possibility for recourse in case of false positives, generalisation to unknown scenarios and robustness to manipulation.

## License

We provide this repository under the [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. You may share (mirror) and adapt (borrow and alter) this content, providing that you credit this work and don't use it for commercial purposes.

## Cite

Either the ICASSP publication :
```
@inproceedings{afchar2025ai,
  title={AI-Generated Music Detection and its Challenges},
  author={Afchar, Darius and Meseguer-Brocal, Gabriel and Hennequin, Romain},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

or the previous Arxiv version (longer paper with more experiments on calibration and interpretability):
```
@article{afchar2024detecting,
  title={Detecting music deepfakes is easy but actually hard},
  author={Afchar, Darius and Meseguer-Brocal, Gabriel and Hennequin, Romain},
  journal={arXiv preprint arXiv:2405.04181},
  year={2024}
}
```

## Reproducibility instructions

To use the autoencoders, you need to clone the following repo into a `pretrained` folder :
* [Git Musika!](https://github.com/marcoppasini/musika)
* [LAC](https://github.com/hugofloresgarcia/lac), using the pretrained weights found in [VampNet](https://github.com/hugofloresgarcia/vampnet)

Then, in the `utils_encode.py` script in the musika folder, I added a method `encode_audio` to return latent from the model given an audio `wv`, namely copying the method `compress_whole_files` and doing a `return lat` instead of doing a `np.save` in the end.


