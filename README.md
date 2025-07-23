# Custom ANN with TensorFlow from Scratch

A simple artificial neural network **from scratch** using TensorFlow's low-level API (`GradientTape`).  
Includes custom training loops, a custom **Huber loss metric**, and uses the **California Housing** dataset.

## Features
- Manual training loop with `tf.GradientTape`
- L2 regularization
- Custom Huber metric implementation
- Progress bar with `tqdm`

##  Requirements
```bash
pip install tensorflow scikit-learn numpy pandas tqdm
