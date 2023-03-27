using LinearAlgebra
using Distributions

@inline function dm_weights(x, proposals, target)
    return target(x) ./ (sum(pdf.(proposals, Ref(x))) ./ length(proposals))
end

function dm_weights_new(x, proposals, target)
    # return target(x) ./ (sum(pdf.(proposals, Ref(x))) ./ length(proposals))
    return vec(map(target, x)) ./ (sum(pdf.(proposals, Ref(x))) ./ length(proposals))
end
