# Notebooks

Use `phase2_model_screening_lab.ipynb` to compare TinyML-friendly TensorFlow/Keras architectures without copying preprocessing or metric code into notebook cells.

The intended workflow is:

1. Define a Keras model-builder function that accepts `(input_shape, num_classes)`.
2. Call `run_keras_architecture(...)`.
3. Compare accuracy, macro F1, model parameters, TFLite size, and host latency from the generated summary.

The notebook deliberately keeps data loading, splitting, normalization, metrics, and TFLite conversion in `src/training/experiment_lab.py` so collaborators only touch model design.
