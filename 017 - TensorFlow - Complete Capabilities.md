# TensorFlow — Complete Capabilities & Features Guide
> A comprehensive reference covering TensorFlow's full ecosystem, APIs, and use cases

---

## 1. What is TensorFlow?

TensorFlow is Google's open-source end-to-end machine learning platform, first released in 2015. It is the most widely used ML framework for **production deployments** at scale, powering Google Search, Gmail, Google Translate, YouTube, and thousands of enterprise systems globally.

TensorFlow provides a **complete stack** — from data ingestion and model building, all the way through distributed training, experiment tracking, deployment to servers, mobile devices, edge hardware, and browsers.

**Current version:** TensorFlow 2.20 (August 2025)
**Installation:**

```bash
pip install tensorflow          # CPU
pip install tensorflow[and-cuda] # GPU (NVIDIA CUDA)
```

```python
import tensorflow as tf
print(tf.__version__)           # verify installation
print(tf.config.list_physical_devices('GPU'))  # check GPU availability
```

---

## 2. The TensorFlow Architecture

TensorFlow is organised as a layered stack:

```
┌──────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│  Keras (tf.keras) · tf.data · TFX · TensorBoard        │
├──────────────────────────────────────────────────────────┤
│                 TRAINING & EVALUATION                    │
│  tf.GradientTape · tf.distribute · tf.function (XLA)   │
├──────────────────────────────────────────────────────────┤
│               CORE TENSOR OPERATIONS                     │
│  tf.Tensor · tf.Variable · tf.math · tf.linalg         │
├──────────────────────────────────────────────────────────┤
│               HARDWARE ACCELERATION                      │
│  CPU · GPU (CUDA) · TPU · Edge (LiteRT)                 │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Tensors — The Fundamental Data Structure

A **tensor** is the basic unit of data in TensorFlow — a multi-dimensional array with a fixed data type and shape.

```python
import tensorflow as tf
import numpy as np

# Scalar (rank-0 tensor)
scalar = tf.constant(42.0)
print(scalar.shape)   # ()
print(scalar.dtype)   # float32

# Vector (rank-1)
vector = tf.constant([1.0, 2.0, 3.0])

# Matrix (rank-2)
matrix = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)

# 3D tensor (e.g. batch of images: [batch, height, width])
images = tf.zeros([32, 28, 28])

# 4D tensor (e.g. batch of colour images: [batch, H, W, channels])
rgb_images = tf.zeros([16, 224, 224, 3])

# Tensor operations
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

print(tf.add(a, b))           # element-wise addition
print(tf.matmul(a, b))        # matrix multiplication
print(tf.reduce_mean(a))      # mean of all elements
print(tf.reshape(a, [4]))     # reshape to 1D

# Convert to/from NumPy seamlessly
np_array = a.numpy()
tf_tensor = tf.constant(np_array)
```

### tf.Variable — Trainable Parameters

```python
# Variables hold mutable state — used for model weights
w = tf.Variable(tf.random.normal([3, 3]))
b = tf.Variable(tf.zeros([3]))

# Variables are automatically tracked by gradient tape
w.assign(w + 0.1)      # update in-place
```

---

## 4. Eager Execution vs Graph Execution

### Eager Execution (default in TF2)

Operations run immediately, like standard Python — great for debugging:

```python
x = tf.constant([1.0, 2.0, 3.0])
y = x ** 2              # executes immediately
print(y)                # tf.Tensor([1. 4. 9.], dtype=float32)
```

### Graph Execution with `@tf.function`

Converts Python functions into high-performance computation graphs using **XLA (Accelerated Linear Algebra)** compilation:

```python
@tf.function
def matrix_multiply(a, b):
    return tf.matmul(a, b)

# First call traces the function and compiles the graph
result = matrix_multiply(
    tf.random.normal([1000, 1000]),
    tf.random.normal([1000, 1000])
)
```

**`@tf.function` benefits:**
- Significant speed-up for repeated calls (graph is compiled once)
- Enables optimisations: operation fusion, constant folding
- Required for TPU execution
- Used automatically by `model.fit()` internally

---

## 5. Building Models with Keras

Keras (`tf.keras`) is TensorFlow's official high-level API. It abstracts neural network building into composable layers and provides built-in training loops.

### 5.1 Sequential API

Best for linear, single-input → single-output architectures:

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),          # explicit input shape

    # Convolutional block
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second conv block
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Dense head
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax'),   # 10 classes
], name='image_classifier')

model.summary()     # print architecture and parameter count
```

### 5.2 Functional API

Best for complex architectures — multi-input, multi-output, residual connections, shared layers:

```python
# Multi-input model: combines image + metadata
image_input = tf.keras.Input(shape=(224, 224, 3), name='image')
metadata_input = tf.keras.Input(shape=(16,), name='metadata')

# Image branch
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(image_input)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Metadata branch
y = tf.keras.layers.Dense(32, activation='relu')(metadata_input)

# Merge and classify
merged = tf.keras.layers.Concatenate()([x, y])
merged = tf.keras.layers.Dense(64, activation='relu')(merged)
output = tf.keras.layers.Dense(5, activation='softmax', name='category')(merged)

model = tf.keras.Model(
    inputs=[image_input, metadata_input],
    outputs=output,
    name='multi_input_model'
)
```

### 5.3 Model Subclassing

Full Python flexibility — custom forward pass logic:

```python
class ResidualBlock(tf.keras.layers.Layer):
    """Custom residual block (skip connection)."""
    def __init__(self, filters):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x, training=training)
        return tf.keras.activations.relu(x + residual)  # skip connection


class CustomResNet(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_init = tf.keras.layers.Conv2D(64, 7, padding='same', activation='relu')
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        x = self.conv_init(x)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.pool(x)
        return self.classifier(x)
```

---

## 6. Layer Reference

### Core Layers

| Layer | Purpose | Key Parameters |
|-------|---------|---------------|
| `Dense` | Fully connected | `units`, `activation`, `kernel_regularizer` |
| `Flatten` | Collapse spatial dims to 1D | — |
| `Embedding` | Map integers to dense vectors | `input_dim`, `output_dim` |
| `Reshape` | Change tensor shape | `target_shape` |

### Convolutional Layers (Images)

| Layer | Purpose | Key Parameters |
|-------|---------|---------------|
| `Conv2D` | 2D spatial convolution | `filters`, `kernel_size`, `strides`, `padding`, `activation` |
| `Conv2DTranspose` | Upsampling (decoder) | same as Conv2D |
| `DepthwiseConv2D` | Lightweight conv (MobileNet-style) | `kernel_size`, `depth_multiplier` |
| `SeparableConv2D` | Depthwise + pointwise | `filters`, `kernel_size` |
| `MaxPooling2D` | Max value in each pool window | `pool_size`, `strides` |
| `AveragePooling2D` | Average value in each window | `pool_size` |
| `GlobalAveragePooling2D` | Spatial → single vector (GAP) | — |

### Recurrent Layers (Sequences)

| Layer | Purpose | Key Parameters |
|-------|---------|---------------|
| `LSTM` | Long Short-Term Memory | `units`, `return_sequences`, `dropout` |
| `GRU` | Gated Recurrent Unit (faster) | `units`, `return_sequences` |
| `SimpleRNN` | Basic RNN (rarely used) | `units` |
| `Bidirectional` | Wrapper for bidirectional RNN | wraps LSTM/GRU |

### Attention & Transformer Layers

| Layer | Purpose |
|-------|---------|
| `MultiHeadAttention` | Scaled dot-product attention |
| `TransformerEncoderLayer` | Encoder block with attention + FFN |

### Regularisation & Normalisation

| Layer | Purpose |
|-------|---------|
| `Dropout` | Random neuron dropout during training |
| `BatchNormalization` | Normalise activations per mini-batch |
| `LayerNormalization` | Normalise per sample (better for RNNs/Transformers) |
| `SpatialDropout2D` | Drop entire feature maps (CNN regularisation) |
| `L1L2` regulariser | Weight penalty via kernel_regularizer |

### Preprocessing Layers (in-model preprocessing)

| Layer | Purpose |
|-------|---------|
| `Normalization` | Subtract mean, divide by std |
| `Rescaling` | Scale pixel values (e.g. /255.0) |
| `RandomFlip` | Data augmentation — random flips |
| `RandomRotation` | Data augmentation — random rotation |
| `RandomZoom` | Data augmentation — random zoom |
| `CategoryEncoding` | One-hot / multi-hot categorical encoding |
| `StringLookup` | Map string categories → integer indices |
| `TextVectorization` | Raw text → token ID sequences |

---

## 7. Activations, Loss Functions, and Optimisers

### Activation Functions

| Function | Equation | When to Use |
|----------|----------|-------------|
| `relu` | max(0, x) | Default for hidden layers |
| `leaky_relu` | max(αx, x), α<1 | Prevents dead ReLU problem |
| `elu` | x if x>0, α(eˣ-1) | Smooth negative region |
| `sigmoid` | 1/(1+e⁻ˣ) | Binary classification output |
| `softmax` | eˣᵢ / Σeˣⱼ | Multiclass output — probabilities sum to 1 |
| `tanh` | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | RNNs, values in (-1, 1) |
| `linear` | x | Regression output |

### Loss Functions

| Loss | Task | Labels |
|------|------|--------|
| `binary_crossentropy` | Binary classification | 0 or 1 |
| `categorical_crossentropy` | Multiclass classification | One-hot vectors |
| `sparse_categorical_crossentropy` | Multiclass classification | Integer class indices |
| `mean_squared_error` | Regression | Continuous values |
| `mean_absolute_error` | Regression (robust) | Continuous values |
| `huber` | Regression (outlier-robust) | Continuous values |
| `kl_divergence` | Distribution matching | Probability distributions |
| `cosine_similarity` | Similarity learning | Normalised vectors |

### Optimisers

| Optimiser | Strengths | Key Hyperparameters |
|-----------|-----------|---------------------|
| `Adam` | Adaptive, robust default | `learning_rate`, `beta_1`, `beta_2`, `epsilon` |
| `SGD` | Simple, good with momentum | `learning_rate`, `momentum`, `nesterov` |
| `RMSprop` | Good for non-stationary | `learning_rate`, `rho`, `epsilon` |
| `Adagrad` | Sparse data, NLP | `learning_rate` |
| `AdamW` | Adam + weight decay | `learning_rate`, `weight_decay` |

---

## 8. Compiling and Training

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
)

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            'best.keras', monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ],
)

# Access training history
import pandas as pd
pd.DataFrame(history.history).plot()  # plot loss and metrics
```

### Custom Training Loop (full control)

```python
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy()
accuracy  = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    accuracy.update_state(y, logits)
    return loss

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
    print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={accuracy.result():.4f}")
    accuracy.reset_state()
```

---

## 9. tf.data — Scalable Input Pipelines

`tf.data` is TensorFlow's high-performance data loading API, designed to feed GPU/TPU training without bottleneck.

```python
import tensorflow as tf

BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

# Build a pipeline from numpy arrays
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=10_000)   # shuffle before batching
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)            # overlap data loading + model training
)

# From TFRecord files (production format, compressed)
def parse_tfrecord(example_proto):
    schema = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, schema)
    image = tf.image.decode_jpeg(parsed['image'], channels=3)
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, parsed['label']

# Read from GCS TFRecord files
dataset = (
    tf.data.TFRecordDataset('gs://my-bucket/data/train-*.tfrecord')
    .map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    .shuffle(5000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
    .cache()                       # cache processed records in memory
)

# Image loading from directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/train/',
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
)
```

### Key tf.data Transformations

| Transformation | Purpose |
|---------------|---------|
| `.shuffle(buffer)` | Randomly shuffle within a buffer |
| `.batch(n)` | Group into batches of size n |
| `.prefetch(AUTOTUNE)` | Overlap data prep with GPU compute |
| `.map(fn, num_parallel_calls)` | Apply a transform to every element in parallel |
| `.filter(fn)` | Keep elements matching a condition |
| `.cache()` | Cache dataset in memory or on disk |
| `.repeat(n)` | Repeat the dataset n times |
| `.interleave(fn)` | Parallel reading from multiple files |
| `.zip(ds1, ds2)` | Merge two datasets element-wise |

---

## 10. Automatic Differentiation — tf.GradientTape

TensorFlow's automatic differentiation engine records operations and computes gradients:

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2 + 2 * x + 1

dy_dx = tape.gradient(y, x)   # computes dy/dx = 2x + 2 = 8.0
print(dy_dx)

# Higher-order gradients
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y = x ** 3
    dy_dx  = inner.gradient(y, x)     # first derivative: 3x²
d2y_dx2 = outer.gradient(dy_dx, x)   # second derivative: 6x
```

---

## 11. Distributed Training — tf.distribute

Scale training across multiple GPUs or machines with minimal code changes:

```python
import tensorflow as tf

# 1. MirroredStrategy — 1 machine, N GPUs (synchronous)
strategy = tf.distribute.MirroredStrategy()
print(f"Replicas: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Scale batch size to make full use of all GPUs
GLOBAL_BATCH = 64 * strategy.num_replicas_in_sync
model.fit(train_dataset, epochs=20)

# 2. MultiWorkerMirroredStrategy — N machines (multi-node)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
# TF_CONFIG environment variable is set by the cluster orchestrator

# 3. TPUStrategy — Cloud TPU pods
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

# 4. ParameterServerStrategy — async, large-scale
strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)
```

### Distribution Strategy Comparison

| Strategy | Replicas | Sync | Best For |
|----------|----------|------|---------|
| `MirroredStrategy` | 1 machine × N GPUs | Synchronous | Standard multi-GPU training |
| `MultiWorkerMirroredStrategy` | N machines × N GPUs | Synchronous | Multi-node clusters |
| `TPUStrategy` | TPU cores | Synchronous | Google Cloud TPUs |
| `ParameterServerStrategy` | N workers + servers | Asynchronous | Very large models, sparse gradients |

---

## 12. Transfer Learning — TensorFlow Hub & Keras Applications

### Keras Applications (built-in pretrained models)

```python
# Load a pretrained base (ImageNet weights)
base = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,         # remove the original classification head
    weights='imagenet',
)
base.trainable = False         # freeze base — only train new head

# Add custom classification head
model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5, activation='softmax'),  # your number of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=10)

# Fine-tune: unfreeze last N layers
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False    # keep early layers frozen

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5))  # lower LR for fine-tuning
model.fit(train_ds, epochs=5)
```

### Available Keras Applications

| Category | Models |
|----------|--------|
| **Image Classification** | VGG16/19, ResNet50/101/152, InceptionV3, EfficientNetB0–B7, MobileNetV2/V3, Xception, NASNetLarge |
| **Object Detection** | Via `tf.keras.applications` + custom heads |
| **Language** | BERT and variants via TensorFlow Hub |

### TensorFlow Hub

```python
import tensorflow_hub as hub

# Load a BERT text embedding
embed = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    trainable=False,
    input_shape=[],
    dtype=tf.string,
)

model = tf.keras.Sequential([
    embed,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax'),
])
```

---

## 13. Model Saving, Exporting, and Loading

```python
# SavedModel format — recommended for production
model.save('my_model/')                  # saves to directory
model.save('gs://bucket/models/v1/')    # directly to GCS

# Load
loaded = tf.keras.models.load_model('my_model/')

# Keras native format (.keras) — TF2.12+
model.save('my_model.keras')

# Weights only (for checkpoint-based workflows)
model.save_weights('checkpoint.weights.h5')
model.load_weights('checkpoint.weights.h5')

# Inspect a SavedModel's signatures
loaded = tf.saved_model.load('my_model/')
print(list(loaded.signatures.keys()))         # ['serving_default']
infer = loaded.signatures['serving_default']
result = infer(tf.constant(x_test[:1]))
```

---

## 14. TensorFlow Ecosystem — Full Product Map

### TensorBoard — Visualisation

```python
# Launch: tensorboard --logdir ./logs --port 6006

callbacks = [tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,       # weight histogram every N epochs
    write_graph=True,       # visualise model graph
    update_freq='epoch',
    profile_batch=2,        # profile second batch for performance
)]

# Custom scalar logging
writer = tf.summary.create_file_writer('./logs/custom')
with writer.as_default():
    for step, value in enumerate(custom_metric_values):
        tf.summary.scalar('my_metric', value, step=step)
```

**TensorBoard tracks:**
- Training / validation loss and metrics per epoch
- Model computation graph
- Weight and bias histograms and distributions
- Embedding projections (UMAP / t-SNE visualisation)
- Custom scalars, images, audio, text, PR curves
- Hardware profiling (GPU/TPU utilisation, memory)

---

### TensorFlow Serving — Production Model Serving

High-performance REST and gRPC serving system for SavedModels:

```bash
# Start the server with Docker
docker run -p 8501:8501 \
  -v /models/my_model:/models/my_model/1 \
  -e MODEL_NAME=my_model \
  tensorflow/serving
```

```python
# Query via REST API
import requests, json
data = json.dumps({'instances': x_test[:3].tolist()})
response = requests.post('http://localhost:8501/v1/models/my_model:predict',
                         data=data)
predictions = response.json()['predictions']
```

**Features:**
- A/B model testing — serve multiple model versions simultaneously
- Canary deployments — gradually shift traffic to new versions
- Automatic batching — group individual requests for GPU efficiency
- Warm model loading — reload without downtime
- Both REST (HTTP/JSON) and gRPC interfaces

---

### TensorFlow Lite (LiteRT) — Edge & Mobile Deployment

Optimised ML inference for devices with limited compute:

```python
# Convert SavedModel → TFLite FlatBuffer
converter = tf.lite.TFLiteConverter.from_saved_model('my_model/')

# Optional: quantise to 8-bit integers (reduces size ~4×, speeds up ~2–3×)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the .tflite file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Run inference with TFLite interpreter
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], x_test[:1])
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])
```

**Quantisation options:**

| Type | Size Reduction | Accuracy Loss | Use Case |
|------|---------------|---------------|---------|
| `DEFAULT` (dynamic range) | ~4× | Minimal | General purpose |
| Full integer (int8) | ~4× | Small | Microcontrollers, DSPs |
| Float16 | ~2× | None | GPU-accelerated edge |
| Post-training quantise-aware | ~4× | Lowest | When accuracy is critical |

**Supported platforms:** Android, iOS, Raspberry Pi, microcontrollers, Linux edge devices

---

### TensorFlow.js — Browser & Node.js ML

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
```

```javascript
// Load a pre-converted model in the browser
const model = await tf.loadLayersModel('/models/my_model/model.json');

// Run inference
const inputTensor = tf.tensor2d([[1.2, 0.8, 3.4, 2.1]]);
const prediction = model.predict(inputTensor);
prediction.print();

// Train directly in the browser
const model = tf.sequential();
model.add(tf.layers.dense({units: 64, activation: 'relu', inputShape: [10]}));
model.add(tf.layers.dense({units: 1}));
model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
await model.fit(xsTrain, ysTrain, {epochs: 20});
```

**Use cases:** In-browser image classification, real-time gesture recognition, client-side NLP, personalized models running locally with no server

---

### TFX — Production ML Pipelines

TensorFlow Extended is an end-to-end pipeline platform for production ML:

```
ExampleGen → StatisticsGen → SchemaGen → ExampleValidator
    ↓
Transform (feature engineering stored as graph)
    ↓
Trainer (model training with TF)
    ↓
Evaluator (model analysis + validation)
    ↓
Pusher (deploy to TF Serving / TFLite / TF.js)
```

**Key TFX components:**

| Component | Library | Purpose |
|-----------|---------|---------|
| ExampleGen | Apache Beam | Ingest data from CSV, BQ, TFRecord |
| StatisticsGen | TFDV | Compute feature statistics |
| SchemaGen | TFDV | Infer schema from statistics |
| ExampleValidator | TFDV | Detect anomalies vs schema |
| Transform | TFT | Feature engineering — consistent train/serve |
| Trainer | TF | Model training |
| Evaluator | TFMA | Model analysis and comparison |
| Pusher | — | Deploy blessed models to serving targets |

**Key advantage:** The `Transform` component stores preprocessing as a TF graph, ensuring **identical transformations** at training time and serving time — eliminating training-serving skew.

---

### TensorFlow Hub — Pre-trained Model Repository

Repository of reusable pre-trained model components:

```python
import tensorflow_hub as hub

# Image feature extraction
feature_extractor = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
    trainable=False,
)

# Text embedding
text_embed = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    trainable=False,
)
```

**Available on TF Hub:** BERT, USE, EfficientNet, MobileNet, YOLO, T5, GPT-2 variants, audio embeddings (YAMNet), video features (I3D)

---

### TensorFlow Data Validation (TFDV)

```python
import tensorflow_data_validation as tfdv

# Compute statistics from a CSV or TFRecord dataset
stats = tfdv.generate_statistics_from_csv('data/train.csv')
tfdv.visualize_statistics(stats)       # visualise in Jupyter

# Infer schema from statistics
schema = tfdv.infer_schema(stats)
tfdv.display_schema(schema)

# Validate a test set against the training schema
test_stats = tfdv.generate_statistics_from_csv('data/test.csv')
anomalies = tfdv.validate_statistics(test_stats, schema)
tfdv.display_anomalies(anomalies)      # detect drift or unexpected values
```

---

### TensorFlow Model Analysis (TFMA)

```python
import tensorflow_model_analysis as tfma

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='label')],
    metrics_specs=[tfma.MetricsSpec(metrics=[
        tfma.MetricConfig(class_name='AUC'),
        tfma.MetricConfig(class_name='BinaryAccuracy'),
    ])],
    slicing_specs=[
        tfma.SlicingSpec(),                       # overall performance
        tfma.SlicingSpec(feature_keys=['gender']),  # performance by gender
        tfma.SlicingSpec(feature_keys=['age_group']),
    ],
)
```

Enables **fairness analysis** — computes metrics across data slices to detect disparities.

---

## 15. Hardware Acceleration

### GPU Training

```python
# TensorFlow automatically uses all available GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

# Limit memory growth (avoid OOM errors)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Explicit device placement
with tf.device('/GPU:0'):
    heavy_computation = tf.matmul(large_matrix_a, large_matrix_b)
```

### Mixed Precision Training (float16/bfloat16)

```python
# ~2× speed-up on modern GPUs (Tensor Cores) and TPUs
tf.keras.mixed_precision.set_global_policy('mixed_float16')

model = build_model()  # computations in float16, weights stored in float32
# Output layer should cast back to float32:
outputs = tf.keras.layers.Dense(10, dtype='float32')(x)
```

### XLA (Accelerated Linear Algebra) Compilation

```python
@tf.function(jit_compile=True)   # compile the entire function with XLA
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_fn(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

---

## 16. TensorFlow vs PyTorch — When to Choose

| Aspect | TensorFlow | PyTorch |
|--------|-----------|---------|
| **Production serving** | Excellent — TF Serving, TFX | Good — TorchServe |
| **Mobile/edge** | Excellent — TFLite / LiteRT | Limited |
| **Browser** | Yes — TF.js | Minimal |
| **Research flexibility** | Good (eager exec) | Excellent (dynamic graphs) |
| **Debugging** | Good in eager mode | Very easy |
| **Community (research papers)** | Large but PyTorch leading | Dominant in research |
| **Enterprise pipelines** | TFX — mature, complete | Less tooling |
| **Learning curve** | Moderate | Lower |
| **Multi-backend Keras** | Yes (Keras 3) | Yes (Keras 3) |

> **Practical rule (2025–2026):** Many teams prototype in PyTorch and deploy with TensorFlow. Keras 3.0's multi-backend support means model code is increasingly portable between both.

---

## 17. TensorFlow Capability Summary

```
CORE
  tf.Tensor / tf.Variable       →  fundamental data structures
  tf.GradientTape               →  automatic differentiation
  @tf.function / XLA            →  graph compilation for speed
  tf.math / tf.linalg           →  mathematical operations

HIGH-LEVEL (KERAS)
  Sequential / Functional / Subclassing  →  model building APIs
  tf.keras.layers               →  100+ built-in layer types
  model.compile / fit / evaluate / predict  →  training lifecycle
  Callbacks                     →  EarlyStopping, Checkpoint, TensorBoard

DATA
  tf.data.Dataset               →  scalable input pipelines
  TFRecord                      →  compressed, production-grade format
  tf.keras.utils.image_dataset_from_directory  →  image loading

TRAINING AT SCALE
  tf.distribute.MirroredStrategy         →  1 machine, N GPUs
  tf.distribute.MultiWorkerMirroredStrategy  →  multi-node
  tf.distribute.TPUStrategy              →  Cloud TPUs
  Mixed precision (float16/bfloat16)     →  2× GPU speed-up

DEPLOYMENT
  TF Serving         →  production REST/gRPC server
  TF Lite / LiteRT   →  mobile, edge, embedded devices
  TF.js              →  browser and Node.js
  SavedModel         →  universal serialisation format

PRODUCTION PIPELINES (TFX)
  ExampleGen → Transform → Trainer → Evaluator → Pusher
  TFDV        →  data validation and drift detection
  TFMA        →  model analysis and fairness slicing
  TensorBoard →  training and experiment visualisation

TRANSFER LEARNING
  tf.keras.applications  →  ImageNet pretrained models
  TensorFlow Hub         →  community pretrained components
```

---

*TensorFlow's unique strength is its **end-to-end production ecosystem** — the same framework handles research prototyping, large-scale distributed training, multi-platform deployment, and continuous pipeline orchestration.*