using LinearAlgebra
using Distributions
using QuasiMonteCarlo

function init_proposals(dim, n_proposals; lims = (-2, 2), σ = 1.0, _rand_ = false)
    proposals = Vector{MvNormal}(undef, n_proposals)
    # plan = randomLHC(n_proposals, dim)
    # plan = scaleLHC(plan, [lims for d in 1:dim])
    if !_rand_
        lb = [lims[1] for i in 1:dim]
        ub = [lims[2] for i in 1:dim]
        plan = QuasiMonteCarlo.sample(n_proposals, lb, ub, LatinHypercubeSample())
        for p_idx = 1:n_proposals
            proposals[p_idx] = MvNormal(plan[:, p_idx], Matrix(σ^2 .* I(dim)))
        end
    else
        width = abs(-(lims...))
        for p_idx = 1:n_proposals
            proposals[p_idx] = MvNormal((rand(dim) .* width) .+ lims[1], Matrix(σ^2 .* I(dim)))
        end
    end
    return proposals
end
