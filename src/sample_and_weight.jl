using Distributions
using Random

include("weights.jl")

function draw_and_weight_samples!(;
    samples::Array,
    weights::Array,
    target::Function,
    proposals::Vector{MvNormal},
    samples_each::Integer,
    current_iteration::Integer,
)

    Threads.@threads for proposal_index = 1:length(proposals)
        _sv = rand(proposals[proposal_index], samples_each)
        samples[current_iteration, proposal_index, :, :] = transpose(_sv)
        # _wv = dm_weights_new(_sv, proposals, target)
        # @info(size(_wv))
        weights[current_iteration, proposal_index, :] = dm_weights_new(_sv, proposals, target)
        # for sample_index = 1:samples_each
        #     samples[current_iteration, proposal_index, sample_index, :] = rand(proposals[proposal_index])
        #     weights[current_iteration, proposal_index, sample_index] = dm_weights(samples[current_iteration, proposal_index, sample_index, :], proposals, target)
        # end
        # @views weights[current_iteration, proposal_index, :] ./ sum(weights[current_iteration, proposal_index, :])
    end
end
