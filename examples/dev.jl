using MLDatasets
using Random
using Flux: unsqueeze
using Plots

using Revise
using ResNet

Random.seed!(0)

include("train.jl")

train_x, train_y = FashionMNIST(; Tx=Float32, split=:train)[1:5000]
test_x, test_y = FashionMNIST(; Tx=Float32, split=:test)[1:1000]

train_x = unsqueeze(train_x, 3)
test_x = unsqueeze(test_x, 3)

indices = randperm(length(test_y))
mid = length(test_y) ÷ 2
val_indices = indices[1:mid]
test_indices = indices[mid + 1:end]

val_x = selectdim(test_x, 4, val_indices)
val_y = test_y[val_indices]

test_x = selectdim(test_x, 4, test_indices)
test_y = test_y[test_indices]

rn = RN(
    block = BasicBlock,
    channels = [4, 4, 4], 
    strides = [2, 2], 
    repeats = [1, 1], 
    grayscale = true,
    classes = 10
)

epochs = 1
(train_losses, train_accs), (val_losses, val_accs) = train!(
    rn, 
    (train_x, train_y), 
    (val_x, val_y), 
    0.01,
    epochs
)

test_acc = accuracy(rn(test_x), test_y)