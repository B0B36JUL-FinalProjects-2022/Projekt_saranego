using MLDatasets
using Random

using Revise
using ResNet

Random.seed!(0)

train_x, train_y = FashionMNIST(; Tx=Float32, split=:train)[:]
test_x, test_y = FashionMNIST(; Tx=Float32, split=:test)[:]

indices = randperm(length(test_y))
mid = length(test_y) ÷ 2
val_indices = indices[1:mid]
test_indices = indices[mid + 1:end]

val_x = test_x[:, :, val_indices]
val_y = test_y[val_indices]

test_x = test_x[:, :, test_indices]
test_y = test_y[test_indices]

RN([1, 4, 8, 16], [2, 2], [2, 2], 10)