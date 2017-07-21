# Indeependence

An autoencoder for deep independence learning and distribution tying. 

## Why?

My goal is to solve catastrophic interference when tying two distributions. 

Deep-mind technique to prevent it uses sensitivity measures as the means to keep track which neurons can be remolded and which should be left alone. In many extends, it has an edge, but also comes with hard parameter tuning and ever increasing stiffness (at a certain point, no new information could be learned). Mathematically, keeping full Fisher information tensors is impractical. Deepmind only approximate it using un-rotated covariance. Research results in machine learning have over and over again suggested that the best way is to keep old data for retraining (through the use of batch or mini-batch). But I do not think that we should go that far to solve this problem. If we only manage to extract certain statistics from the data, that should be enough.

Interestingly, gradients of a function are nothing more than the correlation between the input and the output. Though the size of data may reach the infinity, the dimensionality of correlation is usually much smaller (if it's larger, we should have used rote-learning). When a new data comes, we can compute its correlation and add it up into our statistics.

Momentum method is the first resemblance, but it is not the same thing. Momentum smooth the gradients over optimization steps, while my suggestion tries to smooth over examples. But it is not for all models that we can efficiently keep the statistics; some require infinite amount of storage (like conventional non-linear neural networks). Some are bounded, like models with linear piecewise activation functions, but the alternative of keeping the example data themselves is more attainable.

The only model that we can keep statistics efficiently is the auto-encoder, and that is enough for me...


## Approach

tl,dr; non-linear autoencoder with ReLU mirroring bases.
Please refer to [my blog](https://pvirie.wordpress.com/2017/06/28/deep-independence-learning/) for more detail.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* TensorFlow r1.0
* OpenCV
* numpy
* matplotlib

### Usage

```
usage: main.py [-h] [--load] [--coeff COEFF] [--rate RATE] [--skip SKIP]
               [--limit LIMIT] [--infer INFER]

optional arguments:
  -h, --help     show this help message and exit
  --load         load weight
  --coeff COEFF  update rate
  --rate RATE    learning rate
  --skip SKIP    example skip
  --limit LIMIT  example limit
  --infer INFER  total inference steps
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details