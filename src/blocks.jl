export Block, BasicBlock

abstract type Block end

struct BasicBlock <: Block
    chain::Chain
    channels::Pair{Integer, Integer}
    stride::Integer
end

Flux.@functor BasicBlock

function BasicBlock(channels::Pair{T, T}, stride::Integer, connection::Function) where {T <: Integer}
    before_connection = Chain(
        Conv((3, 3), channels; stride, pad = 1, bias = false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2] => channels[2]; pad = 1, bias = false),
        BatchNorm(channels[2]),
    )
    chain = Chain(SkipConnection(before_connection, connection), relu)
    BasicBlock(chain, channels, stride)
end

(bb::BasicBlock)(x) = bb.chain(x)

Base.show(io::IO, bb::BasicBlock) = print(io, typeof(bb), "(", bb.channels[1], " => ", bb.channels[2], ", ", bb.stride, ")")