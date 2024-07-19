# Benchmarking Neural Network Architectures for Environmental Sound Classification

This study compares various neural network architectures using the ESC-10 dataset for environmental sound classification. 
For more details, see the `Paper.pdf` file.
## Dataset

The ESC-10 dataset, a simplified subset of [ESC-50](https://github.com/karolpiczak/ESC-50), contains 400 clips across 10 classes, with 40 clips per class. This dataset includes a mix of transient/percussive sounds (e.g., sneezing, dog barking), harmonic sounds (e.g., crying baby, crowing rooster), and structured noise/soundscapes (e.g., rain, sea waves).

## Preprocessing and Data Augmentation

### Preprocessing

**Mel Spectrograms:**
- **Audio Transformation:** Convert raw audio signals into Mel Spectrograms.
- **Spectrogram Parameters:**
  - **Sampling Rate:** 16 kHz
  - **Mel Filterbanks:** 124
  - **Window Size:** 1024 samples
  - **Overlap:** 512 samples (50% overlap)
  - **Time Frames:** 156 (32 ms each)

**Logarithmic Power Spectrum and Rescaling:**
- Convert intensities to a logarithmic scale and normalize them ([-1,1]).

### Data Augmentation

Due to the small dataset size, augmentation is performed using the Audiomentations library:
- **Techniques:** 
  - **TimeStretch:** Rate between 0.8 and 1.2
  - **PitchShift:** Semitones between -4 and 4
  - **Shift:** Between -0.1 and 0.1
- **Augmentation Application:** Each sample is augmented three times, quadrupling the training set size.

## Train-Test Folds

5-fold cross-validation is used to train on 4 folds and test on the remaining fold. Only augmented data is used for training, while original data is used for testing. Results from each fold are aggregated for robust evaluation.

## Results

### Averaged Validation Accuracy and Trainable Parameters

| Model             | Validation Accuracy   | Trainable Params |
|-------------------|-----------------------|------------------|
| FNN               | 0.53 ± 0.06           | 24,705,194       |
| CNN               | 0.73 ± 0.08           | 1,257,198        |
| CNN (GlobalPooling)| 0.80 ± 0.01          | 41,410           |
| LSTM              | 0.69 ± 0.04           | 36,870,986       |
| CNN-RF-LSTM       | 0.80 ± 0.05           | 640,202          |
| CNN-TD-LSTM       | 0.76 ± 0.06           | 557,714          |
| LSTM-CNN          | 0.68 ± 0.06           | 203,898          |
| CNN AE            | /                     | 161,905          |
| CNN AE clf        | 0.70 ± 0.06           | 1,656,458        |
| GRU AE            | /                     | 889,600          |
| GRU AE clf        | 0.65 ± 0.02           | 25,418           |


