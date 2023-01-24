module ResNet

export RN

using Flux

struct RN
    entry::Chain
    layers::Chain
    pooling::Chain
    fully_connected::Chain
    activation::Chain
end

(m::RN)(x) = m.activation(m.fully_connected(m.pooling(m.layers(m.entry(x)))))

Flux.@functor RN

function create_basic_block(channels, stride, connection)
    before_connection = Chain(
        Conv((3, 3), channels, stride, pad = 1, bias = false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2] => channels[2], pad = 1, bias = false),
        BatchNorm(channels[2]),
    )
    Chain(SkipConnection(before_connection, connection), relu)
end

function create_layer(block, channels, stride, repeat)
    blocks = []

    if stride == 1 && channels[1] == channels[2]
        push!(blocks, block(channels, +, stride))
    else
        downsample = Chain(
            Conv((1, 1), channels, stride, bias = false),
            BatchNorm(channels[2])
        )
        push!(blocks, block(channels, stride, (x_out, x_in) -> x_out + downsample(x_in)))
    end

    for _ in 2:repeat
        push!(blocks, block(channels[2] => channels[2], 1, +))
    end

    Chain(blocks...)
end

function RN(
    channels::Vector, 
    strides::Vector, 
    repeats::Vector, 
    classes::Integer
    )

    length(channels) - length(strides) == 2 || throw(DomainError(length(channels) - length(strides), "The number of channels must be 2 more than the number of strides"))
    length(strides) == length(repeats) || throw(DomainError(length(strides) - length(repeats), "The number of strides must be the same as the number of repeats"))

    entry = Chain(
        Conv((3, 3), channels[1] => channels[2], pad = 1, bias = false),
        BatchNorm(8, relu)
    )

    layers = []
    in_channels = channels[2]
    
    for (out_channels, stride, repeat) in zip(channels[3:end], strides, repeats)
        push!(layers, create_layer(create_basic_block, in_channels => out_channels, stride, repeat))
        in_channels = out_channels
    end
    layers = Chain(layers...)

    pooling = Chain(
        AdaptiveMeanPool((7, 7)),
        Flux.flatten
    )

    fully_connected = Chain(
        Dense(7 * 7 * channels[end] => classes, bias = false),
        BatchNorm(classes, relu)
    )

    activation = Chain(
        logsoftmax
    )

    RN(entry, layers, pooling, fully_connected, activation)
end

end