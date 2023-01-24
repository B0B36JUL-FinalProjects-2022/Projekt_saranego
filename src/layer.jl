struct Layer
    blocks::Vector{Block}
end

function Layer(block, channels, stride, repeat)
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

    Layer(Chain(blocks...))
end

(l::Layer)(x) = apply_layers(l.blocks, x)

Flux.@functor Layer