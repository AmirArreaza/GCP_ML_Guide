# 🔷 TensorFlow — GCP ML Professional Certification
## Study Guide: TL;DR Reference with Code Snippets

> **Exam Priority:** 🔴 Must Know | 🟡 Important | 🟢 Good to Know

---

## 📌 Quick Reference Table

| Topic | Priority | Core Concept |
|---|---|---|
| `tf.data` Pipelines | 🔴 | Efficient data loading and preprocessing |
| Keras APIs (Sequential, Functional, Subclassing) | 🔴 | Model building approaches |
| Distribution Strategies | 🔴 | Multi-GPU/TPU/multi-worker training |
| TFX Pipeline Components | 🔴 | Production ML pipeline orchestration |
| SavedModel & Serialization | 🔴 | Save, load, and export models |
| Callbacks | 🔴 | Control training behavior |
| GradientTape | 🟡 | Custom training loops |
| Transfer Learning | 🟡 | Reuse pretrained models |
| Preprocessing Layers | 🟡 | In-graph feature engineering |
| TF Serving | 🟡 | Serve models in production |
| Keras Tuner | 🟢 | Hyperparameter search |
| Custom Layers & Losses | 🟢 | Extend Keras with custom logic |
| Regularization Techniques | 🟢 | Prevent overfitting |

---

# 🔴 MUST KNOW

---

## 1. `tf.data` — Input Pipelines

### TL;DR
> *"Never let your GPU wait for data."* `tf.data` builds fast, scalable input pipelines that load, transform, and feed data to your model efficiently.

### Key Methods

| Method | What it does |
|---|---|
| `.map(fn)` | Apply a transformation function to each element |
| `.batch(n)` | Group elements into batches of size n |
| `.shuffle(buffer)` | Randomly shuffle elements (buffer_size matters!) |
| `.prefetch(n)` | Preload next batch while current batch trains |
| `.cache()` | Cache dataset in memory or disk after first epoch |
| `.repeat(n)` | Repeat dataset n times (omit for infinite) |
| `.filter(fn)` | Keep only elements where fn returns True |

### Performance Pattern (Always Use This Order)
```python
dataset = (
    tf.data.Dataset.from_tensor_slices((X, y))
    .shuffle(buffer_size=1000)       # shuffle BEFORE batch
    .map(preprocess_fn,              # transform elements
         num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)                       # then batch
    .cache()                         # cache after expensive ops
    .prefetch(tf.data.AUTOTUNE)      # overlap data prep + training
)
```

### Loading from CSV
```python
dataset = tf.data.experimental.make_csv_dataset(
    "data.csv",
    batch_size=32,
    label_name="target",
    num_epochs=1
)
```

### Loading from TFRecord (Exam Favourite!)
```python
# TFRecords = TensorFlow's binary format, fast for large datasets
raw_dataset = tf.data.TFRecordDataset("data.tfrecord")

feature_description = {
    "feature1": tf.io.FixedLenFeature([], tf.float32),
    "label":    tf.io.FixedLenFeature([], tf.int64),
}

def parse_fn(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

dataset = raw_dataset.map(parse_fn).batch(32).prefetch(tf.data.AUTOTUNE)
```

### 📝 Exam Tips
- Always use `.prefetch(tf.data.AUTOTUNE)` — eliminates CPU/GPU bottleneck
- `buffer_size` in `.shuffle()` affects randomness quality — set to dataset size for full shuffle
- **TFRecord** is the preferred format for large-scale training on GCP (GCS-friendly)
- `tf.data` vs Keras generators → `tf.data` is always faster and more scalable

---

## 2. Keras APIs — Model Building

### TL;DR
> *"Three ways to build a model — pick the right one for the job."*

---

### 2a. Sequential API
> Simple, linear stack of layers. One input → one output, no branching.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

✅ Use when: Simple feedforward networks, quick prototyping  
❌ Avoid when: Multiple inputs/outputs, shared layers, residual connections

---

### 2b. Functional API
> Build models as a graph of layers. Supports branching, merging, and multiple I/O.

```python
# Multi-input model example
input_a = tf.keras.Input(shape=(64,), name="text_features")
input_b = tf.keras.Input(shape=(10,), name="numeric_features")

x = tf.keras.layers.Dense(32, activation='relu')(input_a)
y = tf.keras.layers.Dense(16, activation='relu')(input_b)

merged = tf.keras.layers.Concatenate()([x, y])
output = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
```

✅ Use when: Multiple inputs/outputs, residual/skip connections, shared layers  
✅ **Most common pattern in GCP ML exam scenarios**

---

### 2c. Model Subclassing
> Full Python flexibility — define forward pass in `call()`.

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)  # only active during training
        return self.dense2(x)

model = MyModel()
```

✅ Use when: Research, custom forward pass logic, dynamic architectures  
❌ Avoid when: You need simple deployment/serialization (SavedModel export is trickier)

---

### API Comparison

| Feature | Sequential | Functional | Subclassing |
|---|---|---|---|
| Multiple inputs | ❌ | ✅ | ✅ |
| Multiple outputs | ❌ | ✅ | ✅ |
| Shared layers | ❌ | ✅ | ✅ |
| Residual connections | ❌ | ✅ | ✅ |
| Easiest to debug | ✅ | ✅ | ❌ |
| Best for production | ✅ | ✅ | ⚠️ |
| Full custom logic | ❌ | ❌ | ✅ |

---

## 3. Distribution Strategies

### TL;DR
> *"Scale training across GPUs, machines, or TPUs with one line of code change."*

All strategies wrap your model build + compile inside a `strategy.scope()` context.

---

### 3a. MirroredStrategy — Multi-GPU, Single Machine
```python
strategy = tf.distribute.MirroredStrategy()
# Automatically uses all available GPUs on the machine

with strategy.scope():
    model = tf.keras.Sequential([...])
    model.compile(optimizer='adam', loss='mse')

model.fit(dataset, epochs=10)
```
✅ Use when: One machine with multiple GPUs (e.g., Vertex AI custom training with n1-standard + 4xT4)

---

### 3b. MultiWorkerMirroredStrategy — Multi-Machine, Multi-GPU
```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
```
- Requires a `TF_CONFIG` environment variable on each worker machine
- Vertex AI Training sets `TF_CONFIG` automatically for distributed jobs
✅ Use when: Dataset too large for one machine, need faster training at scale

---

### 3c. TPUStrategy — Cloud TPU Training
```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```
✅ Use when: Very large models or datasets, maximum training speed on GCP  
📝 TPUs require `tf.data` pipelines (no NumPy arrays directly)

---

### Strategy Comparison

| Strategy | Hardware | When to Use |
|---|---|---|
| `MirroredStrategy` | Multi-GPU, 1 machine | Default for GPU training |
| `MultiWorkerMirroredStrategy` | Multi-GPU, multi-machine | Large scale distributed training |
| `TPUStrategy` | Cloud TPU | Fastest training, large models |
| `ParameterServerStrategy` | Multi-machine (async) | Very large models with sparse updates |
| `OneDeviceStrategy` | Single GPU/CPU | Debugging (same API, 1 device) |

### 📝 Exam Tips
- All strategies use the same `strategy.scope()` pattern
- Effective batch size = `batch_size × num_replicas` — **scale learning rate accordingly**
- Vertex AI Training supports all strategies natively

---

## 4. TFX — TensorFlow Extended

### TL;DR
> *"TFX is the production-grade ML pipeline framework for TensorFlow — handles everything from raw data to deployed model."*

### TFX Pipeline Components (Know These!)

```
Raw Data
   ↓
[ExampleGen]      → Ingests data, splits into train/eval
   ↓
[StatisticsGen]   → Computes dataset statistics
   ↓
[SchemaGen]       → Infers data schema (types, ranges)
   ↓
[ExampleValidator]→ Detects anomalies vs. schema
   ↓
[Transform]       → Feature engineering (saved as TF graph)
   ↓
[Trainer]         → Trains the model
   ↓
[Evaluator]       → Evaluates vs. baseline, computes metrics
   ↓
[Pusher]          → Deploys model if evaluation passes
```

### ExampleGen — Ingest Data
```python
from tfx.components import CsvExampleGen

example_gen = CsvExampleGen(input_base='/data/raw')
```

### Transform — Feature Engineering (Runs at Serving Too!)
```python
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    return {
        'scaled_fare': tft.scale_to_z_score(inputs['fare']),
        'vocab_trip_type': tft.compute_and_apply_vocabulary(inputs['trip_type'])
    }
```
📝 **Key exam concept:** `tf.Transform` saves the transformation as a TF graph — the **exact same preprocessing** runs at both training and serving, eliminating training-serving skew.

### Trainer Component
```python
from tfx.components import Trainer
from tfx.proto import trainer_pb2

trainer = Trainer(
    module_file='my_trainer.py',   # contains run_fn()
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=1000),
    eval_args=trainer_pb2.EvalArgs(num_steps=200)
)
```

### Evaluator — Blessed or Not?
```python
from tfx.components import Evaluator
import tensorflow_model_analysis as tfma

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='label')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[tfma.MetricsSpec(metrics=[
        tfma.MetricConfig(class_name='AUC')
    ])]
)

evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    eval_config=eval_config
)
# Output: evaluator.outputs['blessing'] → Pusher checks this
```

### 📝 Exam Tips
- **Transform** is the most exam-tested component — it eliminates training-serving skew
- **Evaluator** uses TFMA (TensorFlow Model Analysis) for sliced evaluation
- TFX pipelines run on **Vertex AI Pipelines** in GCP
- The `Pusher` component only deploys if the model is "blessed" by Evaluator

---

## 5. SavedModel & Serialization

### TL;DR
> *"SavedModel is the universal format — use it for everything in production."*

### Saving a Model
```python
# ✅ Recommended: SavedModel format (default in TF2)
model.save('my_model/')                        # saves as SavedModel dir
model.save('my_model.keras')                   # Keras v3 format

# Legacy h5 format (avoid for new projects)
model.save('my_model.h5')
```

### Loading a Model
```python
model = tf.keras.models.load_model('my_model/')

# For inference only (faster, no training overhead)
loaded = tf.saved_model.load('my_model/')
infer = loaded.signatures['serving_default']
result = infer(tf.constant(input_data))
```

### Saving/Loading Weights Only
```python
model.save_weights('weights/ckpt')
model.load_weights('weights/ckpt')
```

### Inspecting a SavedModel
```bash
# CLI tool to inspect exported SavedModels
saved_model_cli show --dir my_model/ --all
```

### Export for Vertex AI Prediction
```python
# SavedModel must include a serving signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
def serve(x):
    return {'output': model(x)}

tf.saved_model.save(model, 'gs://my-bucket/model/', signatures={'serving_default': serve})
```

### 📝 Exam Tips
- **SavedModel** = the only format fully supported by TF Serving and Vertex AI
- SavedModel saves: architecture + weights + computation graph + signatures
- `.h5` saves architecture + weights but NOT the TF computation graph
- Always export to **GCS** for Vertex AI deployment

---

## 6. Callbacks

### TL;DR
> *"Callbacks let you hook into the training loop — stop early, save checkpoints, adjust learning rate, log to TensorBoard."*

### Most Important Callbacks

```python
callbacks = [

    # Stop training when val_loss stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,               # wait 5 epochs before stopping
        restore_best_weights=True # revert to best model weights
    ),

    # Save the best model during training
    tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/model_{epoch:02d}_{val_loss:.3f}.keras',
        monitor='val_loss',
        save_best_only=True
    ),

    # Reduce LR when metric plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,      # new_lr = lr * factor
        patience=3,
        min_lr=1e-6
    ),

    # Log metrics to TensorBoard
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        update_freq='epoch'
    ),

    # Schedule learning rate manually
    tf.keras.callbacks.LearningRateScheduler(
        schedule=lambda epoch, lr: lr * 0.95 ** epoch
    ),

    # Custom callback
    tf.keras.callbacks.CSVLogger('training_log.csv')
]

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=100,
          callbacks=callbacks)   # pass the list here
```

### Custom Callback
```python
class LogLRCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        print(f"\nEpoch {epoch}: LR = {lr:.6f}, val_loss = {logs['val_loss']:.4f}")
```

### 📝 Exam Tips
- `EarlyStopping` with `restore_best_weights=True` → always use this combo
- `ModelCheckpoint` → saves to GCS path for Vertex AI training jobs
- `TensorBoard` callback → integrates with Vertex AI TensorBoard

---

# 🟡 IMPORTANT

---

## 7. GradientTape — Custom Training Loops

### TL;DR
> *"When `model.fit()` isn't flexible enough, take manual control of the gradient computation."*

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function  # compile to graph for speed
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)

    # Compute gradients of loss w.r.t. trainable weights
    gradients = tape.gradient(loss, model.trainable_variables)

    # Apply gradients to update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_acc.update_state(y_batch, predictions)
    return loss

# Training loop
for epoch in range(10):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)

    print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={train_acc.result():.4f}")
    train_acc.reset_state()
```

### When to Use GradientTape
| Scenario | Use |
|---|---|
| Standard training | `model.fit()` |
| Custom loss combining multiple outputs | `GradientTape` |
| GAN training (generator + discriminator) | `GradientTape` |
| Gradient penalty / clipping | `GradientTape` |
| Meta-learning / MAML | `GradientTape` (nested) |

### Gradient Clipping
```python
gradients = tape.gradient(loss, model.trainable_variables)

# Prevent exploding gradients (important for RNNs)
gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

---

## 8. Transfer Learning & Fine-Tuning

### TL;DR
> *"Don't train from scratch — borrow knowledge from a pretrained model and adapt it to your task."*

### Phase 1 — Feature Extraction (Freeze base)
```python
# Load pretrained base (no top classification layer)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,    # remove ImageNet classifier
    weights='imagenet'    # use pretrained weights
)

# Freeze all base model layers — do NOT update during training
base_model.trainable = False

# Add your own classification head
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # training=False → BatchNorm in inference mode
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)  # 5 custom classes

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

### Phase 2 — Fine-Tuning (Unfreeze top layers)
```python
# After feature extraction converges, unfreeze top layers of base
base_model.trainable = True

# Only fine-tune the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Use a MUCH lower learning rate to avoid destroying pretrained weights
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, epochs=5, validation_data=val_dataset)
```

### Available Keras Pretrained Models
| Model | Size | Best For |
|---|---|---|
| `MobileNetV2` | Small | Mobile, edge, fast inference |
| `EfficientNetB0-B7` | Variable | Best accuracy/size tradeoff |
| `ResNet50` | Medium | General image classification |
| `InceptionV3` | Medium | Complex image patterns |
| `VGG16` | Large | Simple baseline |

### 📝 Exam Tips
- Always use a **lower LR** during fine-tuning vs. feature extraction
- Set `training=False` when calling frozen base model (keeps BatchNorm frozen)
- Fine-tuning = unfreeze top layers only — unfreezing all layers destroys pretrained knowledge

---

## 9. Preprocessing Layers

### TL;DR
> *"Put preprocessing INSIDE the model graph — same transforms guaranteed at training and serving."*

### Why Inside the Model?
```
❌ Bad: Preprocess outside model → risk of training-serving skew
✅ Good: Preprocess inside model → preprocessing is part of the SavedModel
```

### Common Preprocessing Layers
```python
# Normalize numeric features (computes mean/variance from data)
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(X_train_numeric)  # compute stats from training data

# Encode categorical strings to integers + one-hot
vectorizer = tf.keras.layers.StringLookup(output_mode='one_hot')
vectorizer.adapt(train_categories)

# Tokenize and pad text
text_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=100
)
text_vectorizer.adapt(train_texts)

# Resize and rescale images
rescale = tf.keras.layers.Rescaling(scale=1./255)
resize = tf.keras.layers.Resizing(224, 224)
```

### Full Example — Preprocessing Inside Model
```python
# Numeric input branch
num_input = tf.keras.Input(shape=(5,), name='numeric')
num_normalized = normalizer(num_input)

# Text input branch
text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
text_encoded = text_vectorizer(text_input)

# Merge and classify
merged = tf.keras.layers.Concatenate()([num_normalized, text_encoded])
output = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

model = tf.keras.Model(inputs=[num_input, text_input], outputs=output)
# Now model.save() includes ALL preprocessing — safe to deploy!
```

### Data Augmentation (Inside Model for Training Only)
```python
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

# Augmentation only runs when training=True
x = augment(inputs, training=True)
```

---

## 10. TF Serving

### TL;DR
> *"TF Serving is the production server for SavedModels — handles versioning, batching, and REST/gRPC endpoints."*

### Serving a Model (Docker)
```bash
# Pull TF Serving image
docker pull tensorflow/serving

# Serve a SavedModel
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/my_model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  tensorflow/serving
```

### REST Prediction Request
```bash
curl -X POST http://localhost:8501/v1/models/my_model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'
```

### Model Versioning
```
models/
  my_model/
    1/          ← version 1 (older)
      saved_model.pb
      variables/
    2/          ← version 2 (latest, served by default)
      saved_model.pb
      variables/
```

### Deploying on Vertex AI Prediction
```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

# Upload model to Vertex AI Model Registry
model = aiplatform.Model.upload(
    display_name='my-tf-model',
    artifact_uri='gs://my-bucket/my_model/',   # SavedModel in GCS
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest'
)

# Deploy to an endpoint
endpoint = model.deploy(
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=5   # autoscales up to 5 replicas
)

# Predict
endpoint.predict(instances=[[1.0, 2.0, 3.0, 4.0]])
```

### 📝 Exam Tips
- Vertex AI Prediction uses TF Serving internally for TF models
- Model must be in **GCS** as a **SavedModel** before deploying to Vertex AI
- Use `serving_default` signature or define a custom one with `@tf.function`

---

# 🟢 GOOD TO KNOW

---

## 11. Keras Tuner — Hyperparameter Search

### TL;DR
> *"Automatically find the best hyperparameters — learning rate, number of layers, units — without manual guessing."*

```python
import keras_tuner as kt

def build_model(hp):
    model = tf.keras.Sequential()
    # Search over number of units: 32, 64, 128, or 256
    model.add(tf.keras.layers.Dense(
        units=hp.Choice('units', [32, 64, 128, 256]),
        activation='relu'
    ))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Hyperband: efficient early-stopping based search
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    directory='tuner_results',
    project_name='my_search'
)

tuner.search(X_train, y_train, validation_data=(X_val, y_val))

best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(1)[0]
print(f"Best units: {best_hps.get('units')}, Best LR: {best_hps.get('lr')}")
```

### Search Algorithms
| Algorithm | TL;DR |
|---|---|
| `RandomSearch` | Random sampling — simple baseline |
| `Hyperband` | Trains many configs briefly, promotes best ones |
| `BayesianOptimization` | Uses past results to predict best next config |

### 📝 Exam Tip
- Vertex AI **Vizier** is the GCP-managed equivalent of Keras Tuner
- In Vertex AI Training, pass `hyperparameter_tuning_spec` to run Vizier-powered search

---

## 12. Custom Layers & Loss Functions

### TL;DR
> *"When built-in layers and losses don't fit, extend Keras with your own."*

### Custom Layer
```python
class L2NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        # Required for model serialization (save/load)
        config = super().get_config()
        config.update({'axis': self.axis})
        return config
```

### Custom Loss Function
```python
# Simple function-style
def weighted_bce(y_true, y_pred, weight=2.0):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Penalize false negatives more (e.g., medical diagnosis)
    weights = y_true * weight + (1 - y_true)
    return tf.reduce_mean(weights * bce)

# Class-style (required for saving with get_config)
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return tf.reduce_mean((1 - pt) ** self.gamma * bce)

    def get_config(self):
        return {'gamma': self.gamma}
```

### Custom Metric
```python
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        p, r = self.precision.result(), self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
```

---

## 13. Regularization Techniques

### TL;DR
> *"Prevent your model from memorizing training data — generalize better to new data."*

### L1 & L2 Regularization
```python
# L2 (Ridge) — penalizes large weights, most common
tf.keras.layers.Dense(64, activation='relu',
    kernel_regularizer=tf.keras.regularizers.L2(0.01))

# L1 (Lasso) — drives some weights to exactly zero (feature selection)
tf.keras.layers.Dense(64, activation='relu',
    kernel_regularizer=tf.keras.regularizers.L1(0.01))

# L1+L2 (Elastic Net)
tf.keras.layers.Dense(64, activation='relu',
    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))
```

### Dropout
```python
# Randomly zeroes out p% of neurons during training → forces redundancy
tf.keras.layers.Dropout(rate=0.3)  # 30% of neurons dropped

# ⚠️ Dropout is INACTIVE during inference (model(x, training=False))
```

### Batch Normalization
```python
# Normalizes layer inputs → stabilizes training, allows higher LRs
tf.keras.layers.BatchNormalization()

# Best placement: after Dense/Conv, before activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.3)
])
```

### Regularization Cheat Sheet

| Technique | What it does | When to use |
|---|---|---|
| **L2** | Penalizes large weights | Almost always, default choice |
| **L1** | Zeros out unimportant weights | Feature selection needed |
| **Dropout** | Randomly drops neurons | Large networks, dense layers |
| **BatchNorm** | Normalizes activations | Deep networks, unstable training |
| **EarlyStopping** | Stops before overfitting | Always — pair with ModelCheckpoint |
| **Data Augmentation** | Increases training data variety | Image/text tasks with small datasets |

---

## 📊 Exam Quick-Reference Cheat Sheet

| Task | Tool / Approach |
|---|---|
| Fast, scalable data loading | `tf.data` with `.prefetch(AUTOTUNE)` |
| Binary format for large datasets | TFRecord + `TFRecordDataset` |
| Simple model, linear layers | Keras Sequential API |
| Multi-input/output or branching | Keras Functional API |
| Full custom training logic | Model Subclassing + `GradientTape` |
| Multi-GPU single machine | `MirroredStrategy` |
| Multi-machine training | `MultiWorkerMirroredStrategy` |
| Fastest training on GCP | `TPUStrategy` |
| Production ML pipeline | TFX (ExampleGen→Transform→Trainer→Evaluator→Pusher) |
| Eliminate training-serving skew | `tf.Transform` or Keras Preprocessing Layers |
| Save model for production | `model.save()` → SavedModel format |
| Serve model via REST/gRPC | TF Serving or Vertex AI Prediction |
| Stop training automatically | `EarlyStopping` callback |
| Save best model mid-training | `ModelCheckpoint` callback |
| Visualize training | `TensorBoard` callback |
| Reuse pretrained vision model | Keras Applications + `base_model.trainable=False` |
| Automate hyperparameter search | Keras Tuner (or Vertex AI Vizier) |
| Prevent overfitting | Dropout + L2 + EarlyStopping + BatchNorm |

---

*📌 Key exam mindset: The GCP ML exam cares about **when** and **why** to use each tool — not just syntax. Know the tradeoffs: `model.fit()` vs `GradientTape`, Sequential vs Functional, TFRecord vs raw CSV, MirroredStrategy vs TPUStrategy.*