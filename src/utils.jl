export number_of_params

using Flux: params

function apply_layers(layers::Vector, x)
    out = x
    for layer in layers
        out = layer(out)
    end
    out
end

number_of_params(m) = sum(length, params(m))