using LinearAlgebra
using InvertedIndices

@inline function coulomb_repulsion(θ, proposals, p_idx; K = 0.1, self_norm = false)
    # assuming mvnormal
    force = zero(θ)
    _p_det = sqrt(det(proposals[p_idx].Σ))
    for proposal in proposals[Not(p_idx)]
        separation = θ .- proposal.μ
        force .+= separation .* K .* _p_det .* sqrt(det(proposal.Σ)) ./ (norm(separation)^3)
    end
    if self_norm
        force ./= length(proposals)
    end
    return force
end
