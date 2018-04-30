# Model-Compression-Papers
Papers for neural network compression and acceleration. Partly based on [link](https://github.com/memoiry/Awesome-model-compression-and-acceleration/blob/master/README.md).

### Survey

- [Recent Advances in Efficient Computation of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1802.00939.pdf), [arxiv '18]

- [A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/abs/1710.09282) [arXiv '17]


### Quantization

- [The ZipML Framework for Training Models with End-to-End Low Precision: The Cans, the Cannots, and a Little Bit of Deep Learning](https://arxiv.org/abs/1611.05402) [ICML'17]
- [Compressing Deep Convolutional Networks using Vector Quantization](https://arxiv.org/abs/1412.6115) [arXiv'14]
- [Quantized Convolutional Neural Networks for Mobile Devices](https://arxiv.org/abs/1512.06473) [CVPR '16]
- [Fixed-Point Performance Analysis of Recurrent Neural Networks](https://arxiv.org/abs/1512.01322) [ICASSP'16]
- [Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations](https://arxiv.org/abs/1609.07061) [arXiv'16]
- [Loss-aware Binarization of Deep Networks](https://arxiv.org/abs/1611.01600) [ICLR'17]
- [Towards the Limit of Network Quantization](https://arxiv.org/abs/1612.01543) [ICLR'17]
- [Deep Learning with Low Precision by Half-wave Gaussian Quantization](https://arxiv.org/abs/1702.00953) [CVPR'17]
- [ShiftCNN: Generalized Low-Precision Architecture for Inference of Convolutional Neural Networks](https://arxiv.org/abs/1706.02393) [arXiv'17]
- [Training and Inference with Integers in Deep Neural Networks](https://openreview.net/forum?id=HJGXzmspb) [ICLR'18]
- [Deep Learning with Limited Numerical Precision](https://arxiv.org/abs/1502.02551)[ICML'2015]
- [Model compression via distillation and quantization](https://openreview.net/pdf?id=S1XolQbRW) [ICLR '18]
- [Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy](https://openreview.net/pdf?id=B1ae1lZRb) [ICLR '18]
- [On the Universal Approximability of Quantized ReLU Neural Networks](https://arxiv.org/pdf/1802.03646.pdf) [arXiv '18]
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf) [CVPR '18]

### Pruning

- [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) [NIPS'15]
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) [ICLR'17]
- [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) [ICLR'17]
- [Soft Weight-Sharing for Neural Network Compression](https://arxiv.org/abs/1702.04008) [ICLR'17]
- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) [ICLR'16]
- [Dynamic Network Surgery for Efficient DNNs](https://arxiv.org/abs/1608.04493) [NIPS'16]
- [Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning](https://arxiv.org/abs/1611.05128) [CVPR'17]
- [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342) [ICCV'17]
- [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878) [ICLR'18]
- [Data-Driven Sparse Structure Selection for Deep Neural Networks](https://arxiv.org/pdf/1707.01213.pdf)
- [Learning Structured Sparsity in Deep Neural Networks](https://arxiv.org/pdf/1608.03665.pdf) [NIPS '16]
- [Scalpel: Customizing DNN Pruning to the Underlying Hardware Parallelism](http://www-personal.umich.edu/~jiecaoyu/papers/jiecaoyu-isca17.pdf)
- [Learning to Prune: Exploring the Frontier of Fast and Accurate Parsing](http://www.cs.jhu.edu/~jason/papers/vieira+eisner.tacl17.pdf)
- [Channel Pruning for Accelerating Very Deep Neural Networks](https://arxiv.org/pdf/1707.06168.pdf) [ICCV '17]
- [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/pdf/1708.06519.pdf) [ICCV '17]
- [NISP: Pruning Networks using Neuron Importance Score Propagation](https://arxiv.org/pdf/1711.05908.pdf) [CVPR '18]
- [Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://openreview.net/pdf?id=HJ94fqApW) [ICLR '18]
- [MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks](https://arxiv.org/pdf/1711.06798.pdf) [arXiv '17]
- [Efficient Sparse-Winograd Convolutional Neural Networks](https://openreview.net/pdf?id=r1rqJyHKg) [ICLR '18]


### Binarized Neural Network

- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/pdf/1602.02830.pdf)
- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/pdf/1603.05279.pdf)
- [Binarized Convolutional Neural Networks with Separable Filters for Efficient Hardware Acceleration](https://arxiv.org/pdf/1707.04693.pdf)

### Low-rank Approximation

- [Efficient and Accurate Approximations of Nonlinear Convolutional Networks](https://arxiv.org/abs/1411.4229) [CVPR'15]
- [Accelerating Very Deep Convolutional Networks for Classification and Detection](https://arxiv.org/abs/1505.06798) (Extended version of above one)
- [Convolutional neural networks with low-rank regularization](https://arxiv.org/abs/1511.06067) [arXiv'15]
- [Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation](https://arxiv.org/abs/1404.0736) [NIPS'14]
- [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530) [ICLR'16]
- [High performance ultra-low-precision convolutions on mobile devices](https://arxiv.org/abs/1712.02427) [NIPS'17]
- [Speeding up convolutional neural networks with low rank expansions](http://www.robots.ox.ac.uk/~vgg/publications/2014/Jaderberg14b/jaderberg14b.pdf)
- [Coordinating Filters for Faster Deep Neural Networks](https://arxiv.org/pdf/1703.09746.pdf) [ICCV '17]

### Knowledge Distillation

- [Dark knowledge](http://www.ttic.edu/dl/dark14.pdf)
- [FitNets: Hints for Thin Deep Nets](https://arxiv.org/pdf/1412.6550.pdf)
- [Net2net: Accelerating learning via knowledge transfer]()
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [MobileID: Face Model Compression by Distilling Knowledge from Neurons](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11977)
- [DarkRank: Accelerating Deep Metric Learning via Cross Sample Similarities Transfer](https://arxiv.org/pdf/1707.01220.pdf)
- [Deep Model Compression: Distilling Knowledge from Noisy Teachers](https://arxiv.org/pdf/1610.09650.pdf)
- [Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://arxiv.org/pdf/1612.03928.pdf)
- [Sequence-Level Knowledge Distillation](https://arxiv.org/pdf/1606.07947.pdf)
- [Like What You Like: Knowledge Distill via Neuron Selectivity Transfer](https://arxiv.org/pdf/1707.01219.pdf)
- [Learning Efficient Object Detection Models with Knowledge Distillation](http://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf)
- [Data-Free Knowledge Distillation For Deep Neural Networks](https://arxiv.org/pdf/1710.07535.pdf)
- [Learning Loss for Knowledge Distillation with Conditional Adversarial Networks](https://arxiv.org/pdf/1709.00513.pdf)
- [Knowledge Projection for Effective Design of Thinner and Faster Deep Neural Networks](https://arxiv.org/pdf/1710.09505.pdf)
- [Moonshine: Distilling with Cheap Convolutions](https://arxiv.org/pdf/1711.02613.pdf)
- [Model Distillation with Knowledge Transfer from Face Classification to Alignment and Verification](https://arxiv.org/pdf/1709.02929.pdf)
- [Model compression via distillation and quantization](https://openreview.net/pdf?id=S1XolQbRW) [ICLR '18]
- [Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy](https://openreview.net/pdf?id=B1ae1lZRb) [ICLR '18]

### Miscellaneous
- [Beyond Filters: Compact Feature Map for Portable Deep Model](http://proceedings.mlr.press/v70/wang17m/wang17m.pdf) [ICML '17]
- [SplitNet: Learning to Semantically Split Deep Networks for Parameter Reduction and Model Parallelization](http://proceedings.mlr.press/v70/kim17b/kim17b.pdf) [ICML '17]