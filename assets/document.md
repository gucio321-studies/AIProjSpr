# Abstract

Opracowano sieć neuronową wykorzystując podejście uczenia transferowego.
Dokładność klasyfikacji wyniosła

# Goals of the project

The main goal of the project was to develop a neural network capable of classifying electronic components based on their images with high accuracy.

# Theoritical background

## Transfer learning

## Model accuracy

**Accuracy** is a common metric used to evaluate the performance of classification models. It is defined as the ratio of correctly predicted instances to the total number of instances in the dataset. The formula for accuracy is:

$$
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
$$

```{admonition} Random guessing
:class: tip

In a classification problem with `n` classes, random guessing would yield an accuracy of approximately `1/n`.
For example, if there are `10` classes, random guessing would result in an accuracy of about `10%`.
```

## Loss function

As the **accuracy** is deffinitly the most intuitive metric to evaluate the performance of a model, it is not always the best choice for training to optimize against.

Much more commonly used is the **cross-entropy loss** (also known as log loss), which measures the performance of a classification model whose output is a probability value between `0` and `1`. The formula for cross-entropy loss for a single instance is:

$$
\text{Cross-Entropy Loss} = -\sum_{i=1}^{n} y_i \log(p_i)
$$

Where:
- $n$ is the number of classes
- $y_i$ is a binary indicator (0 or 1) if class label `i` is the correct classification for the instance
- $p_i$ is the predicted probability of the instance being classified as class `i`

# Application details

## Dataset preparation

The dataset ([ref 2.](#references)) consists of `10990` images in `34` different categories.

The table below summarizes the number of images in each category:

```{table} dataset_table
| Category | Number of images |
| ---:|:--- |
| solenoid | 317 |
| Integrated-micro-circuit | 470 |
| step-down-transformer | 408 |
| light-circuit | 57 |
| armature | 321 |
| semi-conductor | 421 |
| multiplexer | 85 |
| PNP-transistor | 307 |
| rheostat | 170 |
| semiconductor-diode | 417 |
| junction-transistor | 491 |
| jumper-cable | 249 |
| induction-coil | 151 |
| attenuator | 271 |
| shunt | 104 |
| stabilizer | 162 |
| electric-relay | 420 |
| potential-divider | 237 |
| memory-chip | 297 |
| clip-lead | 288 |
| LED | 480 |
| step-up-transformer | 155 |
| Bypass-capacitor | 302 |
| heat-sink | 420 |
| potentiometer | 468 |
| relay | 443 |
| local-oscillator | 112 |
| pulse-generator | 317 |
| images | 36 |
| omni-directional-antenna | 124 |
| transistor | 153 |
| cartridge-fuse | 225 |
| limiter-clipper | 427 |
| microchip | 461 |
| filament | 400 |
| Electrolytic-capacitor | 403 |
| microprocessor | 457 |
```

```{admonition} Images preprocessing
- In order to save resources, each image have been resized to `128x64` pixels using the OpenCV's Inter Area algorithm (see [ref 3.](#references)).
- The grayscale images have been converted to RGB format (by replacing the single channel with three identical channels).
- The alpha channel of RGBA images have been removed.
```

A random permutation of the dataset was applied to ensure that the training and validation sets are representative of the overall dataset.
The dataset has been split into training and validation sets with a ratio of `80:20`.

# Model architecture

# Training results

# Summary

# References

1. Project source code <https://github.com/>
2. Dataset source <https://www.kaggle.com/datasets/aryaminus/electronic-components>
3. OpenCV's interpolation methods <https://opencv-opencv.mintlify.app/examples/image-transformations#interpolation-methods>
