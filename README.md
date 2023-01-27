# ResNet

Implementation of residual network.

## Usage

To create your own residual network use the constructor `RN()`.
The returned model is a `Flux` functor, so you can train it with your own `Flux` training loop.
Available blocks are defined in the `blocks.jl`.
For detailed information read the documentation of the constructor.

```julia
RN(
    block = BasicBlock,
    channels = [2, 4, 8], 
    strides = [2, 2], 
    repeats = [2, 2], 
    grayscale = true,
    classes = 10
)
```

```
ResNet with 8 layers and 6578 parameters

Entry
    Conv((3, 3), 1 => 2, pad=1, bias=false)
    BatchNorm(2, relu)
Layer1
    BasicBlock(2 => 4, stride=2)
    BasicBlock(4 => 4, stride=1)
Layer2
    BasicBlock(4 => 8, stride=2)
    BasicBlock(8 => 8, stride=1)
Head
    AdaptiveMeanPool((7, 7))
    flatten
    Dense(392 => 10; bias=false)
    BatchNorm(10)
    logsoftmax
```

```julia
RN(
    block = Bottleneck,
    channels = [32, 64, 128, 256], 
    strides = [1, 2, 2], 
    repeats = [4, 4, 4], 
    grayscale = false,
    classes = 20,
    pooling_dims = (5, 5),
    expansion = 4
)
```

```
ResNet with 38 layers and 6205384 parameters

Entry
    Conv((3, 3), 3 => 32, pad=1, bias=false)
    BatchNorm(32, relu)
Layer1
    Bottleneck(32 => 64, stride=1, expansion=4)
    Bottleneck(256 => 64, stride=1, expansion=4)
    Bottleneck(256 => 64, stride=1, expansion=4)
    Bottleneck(256 => 64, stride=1, expansion=4)
Layer2
    Bottleneck(256 => 128, stride=2, expansion=4)
    Bottleneck(512 => 128, stride=1, expansion=4)
    Bottleneck(512 => 128, stride=1, expansion=4)
    Bottleneck(512 => 128, stride=1, expansion=4)
Layer3
    Bottleneck(512 => 256, stride=2, expansion=4)
    Bottleneck(1024 => 256, stride=1, expansion=4)
    Bottleneck(1024 => 256, stride=1, expansion=4)
    Bottleneck(1024 => 256, stride=1, expansion=4)
Head
    AdaptiveMeanPool((5, 5))
    flatten
    Dense(25600 => 20; bias=false)
    BatchNorm(20)
    logsoftmax
```