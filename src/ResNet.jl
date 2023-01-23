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

end