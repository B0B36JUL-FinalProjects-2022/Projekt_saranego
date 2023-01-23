module ResNet

using Flux

function create_basic_block(channels, connection; stride = 1)
    before_connection = Chain(
        Conv((3, 3), channels; stride, pad = 1, bias = false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2] => channels[2]; pad = 1, bias = false),
        BatchNorm(channels[2]),
    )
    Chain(SkipConnection(before_connection, connection), relu)
end

function create_layer(block, size, channels; stride = 1)
    layer = []

    if stride == 1 && channels[1] == channels[2]
        push!(layer, block(channels, +; stride))
    else
        downsample = Chain(
            Conv((1, 1), channels; stride, bias = False),
            BatchNorm(channels[2])
        )
        push!(layer, block(channels, (x_out, x_in) -> x_out + downsample(x_in); stride))
    end

    for _ in 2:size
        push!(layer, block(channels[2] => channels[2], +))
    end

    Chain(layer...)
end

end