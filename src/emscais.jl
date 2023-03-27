using LinearAlgebra
using Distributions
using DrWatson
using Zygote

include("weights.jl")
include("sample_and_weight.jl")
include("init_props.jl")

@inline function hmc4emscais!(
    θ,
    log_target;
    d_log_target = x -> gradient(k -> log_target(k), x)[1],
    n_iterations = 10,
    inv_M = Matrix(I(length(θ))),
    ϵ = 0.1,
    L = 10,
    K = 1.0,
    repulsion = false,
    proposals = nothing,
    p_idx,
)
    # should be faster than calling AdvancedHMC routines as those have overhead to make the samples better
    # not needed here as samples themselves not of interest
    ϕ_dist = MvNormal(zero(θ), inv(inv_M))
    for i = 1:n_iterations
        θ′ = copy(θ)
        ϕ = rand(ϕ_dist)
        ϕ′ = copy(ϕ)
        if repulsion == true
            for leapfrog_step = 1:L
                # @info("log_target", d_log_target(θ′))
                # @info("repulsion", emscais_coulomb_repulsion(θ′, proposals, p_idx))
                ϕ′ .+= 0.5 .* ϵ .* (d_log_target(θ′) .+ K .* coulomb_repulsion(θ′, proposals, p_idx))
                θ′ .+= ϵ .* inv_M * ϕ′
                ϕ′ .+= 0.5 .* ϵ .* (d_log_target(θ′) .+ K .* coulomb_repulsion(θ′, proposals, p_idx))
            end
            # @info("log_target", d_log_target(θ′))
            # @info("repulsion", emscais_coulomb_repulsion(θ′, proposals, p_idx))
        else
            for leapfrog_step = 1:L
                ϕ′ .+= 0.5 .* ϵ .* d_log_target(θ′)
                θ′ .+= ϵ .* inv_M * ϕ′
                ϕ′ .+= 0.5 .* ϵ .* d_log_target(θ′)
            end
        end
        mha = log_target(θ′) + logpdf(ϕ_dist, ϕ′) - log_target(θ) - logpdf(ϕ_dist, ϕ)
        if log(rand()) < mha
            θ = θ′
        else
            # momentum flip
            ϕ .*= -1.0
            θ′ = copy(θ)
            ϕ′ = copy(ϕ)
            if repulsion == true
                for leapfrog_step = 1:L
                    # @info("log_target", d_log_target(θ′))
                    # @info("repulsion", emscais_coulomb_repulsion(θ′, proposals, p_idx))
                    ϕ′ .+= 0.5 .* ϵ .* (d_log_target(θ′) .+ K .* coulomb_repulsion(θ′, proposals, p_idx))
                    θ′ .+= ϵ .* inv_M * ϕ′
                    ϕ′ .+= 0.5 .* ϵ .* (d_log_target(θ′) .+ K .* coulomb_repulsion(θ′, proposals, p_idx))
                end
                # @info("log_target", d_log_target(θ′))
                # @info("repulsion", emscais_coulomb_repulsion(θ′, proposals, p_idx))
            else
                for leapfrog_step = 1:L
                    ϕ′ .+= 0.5 .* ϵ .* d_log_target(θ′)
                    θ′ .+= ϵ .* inv_M * ϕ′
                    ϕ′ .+= 0.5 .* ϵ .* d_log_target(θ′)
                end
            end
            mha = log_target(θ′) + logpdf(ϕ_dist, ϕ′) - log_target(θ) - logpdf(ϕ_dist, ϕ)
            if log(rand()) < mha
                θ = θ′
            end
        end
    end
    return θ
end


function emscais_local_adapt!(
    samples,
    weights,
    transformed_weights;
    samples_each,
    proposals,
    current_iteration,
    N_t,
    γ,
    mdiff_arr,
    cov_arr,
    β = 0.1,
    η = 0.1,
)
    rv_c = zeros(length(proposals), length(proposals[begin]), length(proposals[begin]))
    rv_m = zeros(length(proposals), length(proposals[begin]))
    for proposal_index = 1:length(proposals)
        @views _samples = samples[current_iteration, proposal_index, :, :]
        @views p_samples = [_samples[i, :] for i in 1:samples_each]
        @views _weights = weights[current_iteration, proposal_index, :]
        @views weight_sum = sum(_weights)
        @views square_weight_sum = sum(x -> x^2, _weights)
        @views _ess = inv(square_weight_sum)

        @views norm_weights = _weights ./ weight_sum

        _degen_flag = (_ess <= N_t)

        if _degen_flag
            transformed_weights[current_iteration, proposal_index, :] .= _weights .^ inv(γ)
        else
            transformed_weights[current_iteration, proposal_index, :] .= _weights
        end

        @views _transformed_weights = transformed_weights[current_iteration, proposal_index, :]

        @views t_weight_sum = sum(_transformed_weights)
        @views square_t_weight_sum = sum(x -> x^2, _transformed_weights)

        @views norm_t_weights = _transformed_weights ./ t_weight_sum

        @views local_mean = sum(norm_weights .* p_samples)

        @views mean_diff = p_samples .- [local_mean for i in 1:samples_each]

        # for sample_index = 1:samples_each
        #     cov_arr[proposal_index, sample_index, :, :] =
        #         (_samples[sample_index, :] .- local_mean) * transpose(_samples[sample_index, :] .- local_mean)
        # end

        W_regular = 1.0 .- sum(norm_weights .^ 2)
        W_trans = 1.0 .- sum(norm_t_weights .^ 2)

        @views _cov_arr = [i * i' for i in mean_diff]

        @views is_cov1 = inv(W_regular) .* sum(norm_weights .* _cov_arr)
        @views is_cov2 = inv(W_trans) .* sum(norm_t_weights .* _cov_arr)

        # @views is_cov1 =
        #     inv(W_regular) .*
        #     sum(norm_weights .* cov_arr[proposal_index, :, :, :], dims = 1)[
        #         1,
        #         :,
        #         :,
        #     ]
            # if proposal_index == 1
            #     @info("cov1")
            #     display(is_cov1)
            # end
        #
        # @views is_cov2 =
        #     inv(W_trans) .* sum(
        #         norm_t_weights .* cov_arr[proposal_index, :, :, :],
        #         dims = 1,
        #     )[
        #         1,
        #         :,
        #         :,
        #     ]
            # if proposal_index == 1
            #     @info("cov2")
            #     display(is_cov2)
            # end

        Σ = (1.0 .- β) .* proposals[proposal_index].Σ .+ β .* (1 - η) .* is_cov1 .+ β .* η .* is_cov2
        rv_c[proposal_index, :, :] .= Σ
        rv_m[proposal_index, :] .= local_mean
    end
    return (rv_m, rv_c)
end

function emscais_global_adapt!(log_target; proposals, HMC_args)
    rv_m = zeros(length(proposals), length(proposals[begin]))
    for proposal_index = 1:length(proposals)
        rv_m[proposal_index, :] = hmc4emscais!(proposals[proposal_index].μ, log_target; p_idx = proposal_index, HMC_args...)
    end
    return rv_m
end


function emscais!(
    target,
    samples,
    weights,
    transformed_weights;
    proposals,
    iterations,
    samples_each,
    α = repeat([0.1], iterations),
    β = repeat([0.1], iterations),
    η = repeat([0.1], iterations),
    κ = repeat([0.1], iterations),
    γ = repeat([3.0], iterations),
    N_t = round(Int, 0.1 * samples_each),
    HMC_args = [Dict() for x = 1:iterations],
)
    n_proposals = length(proposals)
    x_dim = length(proposals[begin])
    mdiff_arr = zeros(n_proposals, samples_each, x_dim)
    cov_arr = zeros(n_proposals, samples_each, x_dim, x_dim)

    for iteration = 1:iterations
        @views draw_and_weight_samples!(
            samples = samples,
            weights = weights,
            target = target,
            proposals = proposals,
            current_iteration = iteration,
            samples_each = samples_each,
        )
         μ1, Σ = emscais_local_adapt!(
            samples,
            weights,
            transformed_weights;
            samples_each = samples_each,
            proposals = proposals,
            current_iteration = iteration,
            N_t = N_t,
            γ = γ[iteration],
            mdiff_arr = mdiff_arr,
            cov_arr = cov_arr,
            β = β[iteration],
            η = η[iteration],
        )
        @views μ2 = emscais_global_adapt!(x -> log(target(x)); proposals = proposals, HMC_args = HMC_args[iteration])
        for proposal_index = 1:n_proposals
            @views μ =
                (1.0 .- α[iteration]) .* proposals[proposal_index].μ .+
                α[iteration] .* (1 - κ[iteration]) .* μ1[proposal_index, :] .+
                α[iteration] .* κ[iteration] .* μ2[proposal_index, :]
            # try
                proposals[proposal_index] = MvNormal(μ, Σ[proposal_index, :, :])
            # catch
            #     display(Σ[1, :, :])
            # end
        end
    end
    return (samples, weights)
end
