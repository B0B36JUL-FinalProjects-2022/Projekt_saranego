export Block, BasicBlock

abstract type Block end

struct BasicBlock
    chain::Chain
    channels::Pair{Integer, Integer}
    stride::Integer
end

function BasicBlock(channels, stride, connection)
    before_connection = Chain(
        Conv((3, 3), channels, stride, pad = 1, bias = false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2] => channels[2], pad = 1, bias = false),
        BatchNorm(channels[2]),
    )
    chain = Chain(SkipConnection(before_connection, connection), relu)
    BasicBlock(chain, channels, stride)
end

(bb::BasicBlock)(x) = bb.chain(x)

Flux.@functor BasicBlock