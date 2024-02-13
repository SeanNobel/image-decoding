# Image Decoding

### Unofficial minimal re-implementation of the paper "Brain Decoding: Toward Real-Time Reconstruction of Visual Perception" from FAIR, Meta

## Status

- :construction: Preprocess and CLIP training are implemented but ubable to reproduce the top-5 accuracy reported in the paper.

  - Currently using the preprocessed data in Hebart et al., 2023, whose pipeline is slightly different from the Meta paper.

  - Currently only CLIP-Vision encoding is supported.

- :construction: Generation module to be implemented.

## Usage

- Download the THINGS-MEG dataset from [here](https://openneuro.org/datasets/ds004212/versions/2.0.0) and place it in where you want.

- Run preprocessing.

```bash
python preproc.py data_dir={path to the THINGS-MEG dataset directory with / at the end}
```