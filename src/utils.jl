function apply_layers(layers::Vector, x)
    out = x
    for layer in layers
        out = layer(out)
    end
    out
end