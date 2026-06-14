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

```{table} Dataset distribution
| Category | Number of images |
|---:|:---|
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
| images | 36 (ignored) |
| omni-directional-antenna | 124 |
| transistor | 153 |
| cartridge-fuse | 225 |
| limiter-clipper | 427 |
| microchip | 461 |
| filament | 400 |
| Electrolytic-capacitor | 403 |
| microprocessor | 457 |
```

```{important}
The category `images` is a kind of `kaggle` artifact and in fact does not contain any images (it is gnored in the further training).
```

```{admonition} Images preprocessing
- In order to save resources, each image have been resized to `128x64` pixels using the OpenCV's Inter Area algorithm (see [ref 3.](#references)).
- The grayscale images have been converted to RGB format (by replacing the single channel with three identical channels).
- The alpha channel of RGBA images have been removed.
```

A random permutation of the dataset was applied to ensure that the training and validation sets are representative of the overall dataset.
The dataset has been split into training and validation sets with a ratio of `80:20`.

## Model architecture

There were two approaches considered for the model architecture:
- A Convolutional Neural Network (CNN) trained from scratch.
- A transfer learning approach using a pre-trained models:
    * MobileNetV2
    * ResNet50
    * EfficientNetB0
    * EfficientNetB2

### Convolutional Neural Network (CNN)

The network used in that apprach was made of one convolutional layer with `64` filters of size `3x3`, followed by a max pooling, flatten and dense layers.
The following code listing shows the exact architecture of the CNN with all the parameters:

```python
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))

model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(64, activation='relu'))

# Output layer
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(len(categories), activation='softmax'))

model.summary()
```

### Transfer learning models

## Learning procedure

The following learning procedure was applied to all the models:
1. First-stage training: the base model (pre-trained on ImageNet) was frozen and only the top layers were trained.
There were Early Stopping ass well as Reduce Learning Rate callbacks applied to improve the procedure.
```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=5,             
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks=[early_stopping, reduce_lr]
                    )
```
2. Second-stage - fine tuning: the pre-trained part of a model has been unfrozen, and the training was repeated with a smaller initial learning rate value of `10^{-5}`.
The Early Stopping as well as Reduce Learning Rate callbacks were applied, but this time, they were meant to optimize the validation accuracy instead of the validation loss.
```python
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    mode="max",
      patience=8,             # epochs without improvement
      restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    mode="max",
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history_ft = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=ft_epohs,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr]
)
```

```{important}
The performance of `val_loss` vs `val_accuracy` as an optimization target for the callbacks was not tested and will not be discussed here.
```

```{note}
For CNN, the second stage of training was not applied. Also, no layers were frozen.
```

# Training results

THe learning procedure as described above was applied to all the models. The table below summarizes the results of the training process:

## CNN Network
## MobileNetV2

```{figure} ./mobileNetV2/accuracy_pretune.png
:width: 75%

MobileNetV2 Accuracy and Validation Accuracy before fine tuning.
```

```{figure} ./mobileNetV2/accuracy_posttune.png
:width: 75%

MobileNetV2 Accuracy and Validation Accuracy after fine tuning.
```


```{table}
| Model | Accuracy before tune [%] | Loss before tune | Accuracy after tune [%] | Loss after tune |
|---:|:---:|:---:|:---:|:---:|
| CNN | 19.93 | 2.8599 | N/A | N/A |
| MobileNetV2 | 47.68 | 1.7457 | 46.59 | 1.7837 |
| EfficientNetB0 |  51.14 | 1.5922 | 51.64 | 1.5652 |
| EfficientNetB2 |  51.05 | 1.5954 | 49 | 1.6648 |
| ResNet50 | 50.18 | 1.6864 | 50.55 | 1.738 |
```

## Models Comparison


# Summary

# References

1. Project source code <https://github.com/>
2. Dataset source <https://www.kaggle.com/datasets/aryaminus/electronic-components>
3. OpenCV's interpolation methods <https://opencv-opencv.mintlify.app/examples/image-transformations#interpolation-methods>
