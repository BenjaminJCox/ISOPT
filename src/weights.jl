using LinearAlgebra
using Distributions

@inline function dm_weights(x, proposals, target)
    return target(x) ./ (sum(pdf.(proposals, Ref(x))) ./ length(proposals))
end

function dm_weights_new(x, proposals, target)
    # return target(x) ./ (sum(pdf.(proposals, Ref(x))) ./ length(proposals))
    # @info(size(vec(mapslices(target, x, dims = 1))))
    # @info(size(sum(pdf.(proposals, Ref(x)))))
    _t1 = vec(mapslices(target, x, dims = 1))
    _t2 = sum(pdf.(proposals, Ref(x)))
    _t3 = length(proposals)
    return _t1 ./ (_t2 ./ _t3)
end
