using MLDatasets
using Random
using Flux: unsqueeze

using Revise
using ResNet

Random.seed!(0)

train_x, train_y = FashionMNIST(; Tx=Float32, split=:train)[:]
test_x, test_y = FashionMNIST(; Tx=Float32, split=:test)[:]

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

rn = RN([1, 4, 8, 16], [2, 2], [2, 2], 10)

samples = selectdim(train_x, 4, 1:5)
size(rn(samples))