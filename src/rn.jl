export RN

struct RN
    entry::Chain
    layers::Chain
    head::Chain
end

Flux.@functor RN

function RN(channels::Vector, strides::Vector, repeats::Vector, classes::Integer; pooling_dims = (7, 7))
    length(channels) - length(strides) == 2 || throw(DomainError(length(channels) - length(strides), "The number of channels must be 2 more than the number of strides"))
    length(strides) == length(repeats) || throw(DomainError(length(strides) - length(repeats), "The number of strides must be the same as the number of repeats"))

    entry = Chain(
        Conv((3, 3), channels[1] => channels[2], pad = 1, bias = false),
        BatchNorm(channels[2], relu)
    )

    layers = []
    in_channels = channels[2]
    
    for (out_channels, stride, repeat) in zip(channels[3:end], strides, repeats)
        push!(layers, Layer(BasicBlock, in_channels => out_channels, stride, repeat))
        in_channels = out_channels
    end
    layers = Chain(layers...)

    head = Chain(
        AdaptiveMeanPool(pooling_dims),
        Flux.flatten,
        Dense(prod(pooling_dims) * channels[end] => classes, bias = false),
        BatchNorm(classes),
        logsoftmax
    )

    RN(entry, layers, head)
end

(rn::RN)(x) = rn.head(apply_layers(rn.layers, rn.entry(x)))

function Base.show(io::IO, rn::RN)
    n_layers = length(rn.entry) + length(rn.layers) + length(rn.head)
    println("ResNet with ", n_layers, " layers and ", number_of_params(rn), " parameters\n")
    
    println("EntryLayer")
    for layer in rn.entry
        print(" " ^ 4)
        println(layer)
    end

    for (i, layer) in enumerate(rn.layers)
        println("ResidualLayer", i)

        for block in layer.blocks
            print(" " ^ 4)
            println(block)
        end
    end

    println("HeadLayer")
    for layer in rn.head
        print(" " ^ 4)
        println(layer)
    end
end