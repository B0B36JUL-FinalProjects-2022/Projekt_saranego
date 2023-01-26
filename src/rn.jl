export RN

struct RN
    entry::Chain
    layers::Chain
    head::Chain
end

Flux.@functor RN

function RN(; 
        block,
        channels::Vector, 
        strides::Vector, 
        repeats::Vector, 
        classes::Integer, 
        grayscale = false,
        pooling_dims = (7, 7),
        kwargs...
    )

    length(channels) - length(strides) == 1 || throw(DomainError(length(channels) - length(strides), "The number of channels must be 1 more than the number of strides"))
    length(strides) == length(repeats) || throw(DomainError(length(strides) - length(repeats), "The number of strides must be the same as the number of repeats"))

    entry = Chain(
        Conv((3, 3), (grayscale ? 1 : 3) => channels[1], pad = 1, bias = false),
        BatchNorm(channels[1], relu)
    )

    layers = []
    in_channels = channels[1]
    expansion = get(Dict(kwargs), :expansion, 1)
    
    for (out_channels, stride, repeat) in zip(channels[2:end], strides, repeats)
        push!(layers, Layer(block, in_channels => out_channels, stride, repeat; kwargs...))
        in_channels = out_channels * expansion
    end
    layers = Chain(layers...)

    head = Chain(
        AdaptiveMeanPool(pooling_dims),
        Flux.flatten,
        Dense(prod(pooling_dims) * channels[end] * expansion => classes, bias = false),
        BatchNorm(classes),
        logsoftmax
    )

    RN(entry, layers, head)
end

RN(::BasicBlock, args...; kwargs...) = RN(BasicBlock, args...; kwargs...)
RN(::Bottleneck, args...; kwargs...) = RN(Bottleneck, args...; kwargs...)

(rn::RN)(x) = rn.head(rn.layers(rn.entry(x)))

function Base.show(io::IO, rn::RN)
    n_layers = length(rn.entry) + length(rn.layers) + length(rn.head)
    println(io, "ResNet with ", n_layers, " layers and ", number_of_params(rn), " parameters\n")
    
    println(io, "Entry")
    for layer in rn.entry
        print(io, " " ^ 4)
        println(io, layer)
    end

    for (i, layer) in enumerate(rn.layers)
        println(io, "Layer", i)

        for block in layer.blocks
            print(io, " " ^ 4)
            println(io, block)
        end
    end

    println(io, "Head")
    for layer in rn.head
        print(io, " " ^ 4)
        println(io, layer)
    end
end