using LinearAlgebra
using Distributions
using DrWatson
using Zygote

include("weights.jl")
include("sample_and_weight.jl")
include("init_props.jl")
include("repulsion.jl")

@inline function hmc4emscais!(
    θ,
    log_target;
    d_log_target,
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
        if repulsion == true && K > 0
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
            if repulsion == true && K > 0
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
    cov_adapt = true,
)
    rv_c = zeros(length(proposals), length(proposals[begin]), length(proposals[begin]))
    rv_m = zeros(length(proposals), length(proposals[begin]))
    Threads.@threads for proposal_index = 1:length(proposals)
        # for proposal_index = 1:length(proposals)
        @views _samples = samples[current_iteration, proposal_index, :, :]
        @views p_samples = [_samples[i, :] for i = 1:samples_each]
        @views _weights = weights[current_iteration, proposal_index, :]
        @views weight_sum = sum(_weights)
        @views square_weight_sum = sum(x -> x^2, _weights)
        @views _ess = inv(square_weight_sum)

        @views norm_weights = _weights ./ weight_sum

        _degen_flag = (_ess <= N_t)

        if _degen_flag
            @views transformed_weights[current_iteration, proposal_index, :] .= _weights .^ inv(γ)
        else
            @views transformed_weights[current_iteration, proposal_index, :] .= _weights
        end

        @views _transformed_weights = transformed_weights[current_iteration, proposal_index, :]

        @views t_weight_sum = sum(_transformed_weights)
        @views square_t_weight_sum = sum(x -> x^2, _transformed_weights)

        @views norm_t_weights = _transformed_weights ./ t_weight_sum

        @views local_mean = sum(norm_weights .* p_samples)

        if cov_adapt
            @views mean_diff = p_samples .- [local_mean for i = 1:samples_each]

            # for sample_index = 1:samples_each
            #     cov_arr[proposal_index, sample_index, :, :] =
            #         (_samples[sample_index, :] .- local_mean) * transpose(_samples[sample_index, :] .- local_mean)
            # end

            @views W_regular = 1.0 .- sum(norm_weights .^ 2)
            @views W_trans = 1.0 .- sum(norm_t_weights .^ 2)

            @views _cov_arr = [i * i' for i in mean_diff]

            @views is_cov1 = inv(W_regular) .* sum(norm_weights .* _cov_arr)
            @views is_cov2 = inv(W_trans) .* sum(norm_t_weights .* _cov_arr)

            @views Σ = (1.0 .- β) .* proposals[proposal_index].Σ .+ β .* (1 - η) .* is_cov1 .+ β .* η .* is_cov2
        else
            @views Σ = proposals[proposal_index].Σ
        end

        rv_c[proposal_index, :, :] .= Σ
        rv_m[proposal_index, :] .= local_mean
    end
    return (rv_m, rv_c)
end

function emscais_global_adapt!(log_target; proposals, HMC_args)
    rv_m = zeros(length(proposals), length(proposals[begin]))
    Threads.@threads for proposal_index = 1:length(proposals)
        # for proposal_index = 1:length(proposals)
        rv_m[proposal_index, :] = hmc4emscais!(proposals[proposal_index].μ, log_target; p_idx = proposal_index, proposals = proposals, HMC_args...)
    end
    return rv_m
end


function emscais_gradient_adapt!(target, d_log_target; proposals, attenuator)
    rv_m = zeros(length(proposals), length(proposals[begin]))
    Threads.@threads for proposal_index = 1:length(proposals)
    # for proposal_index = 1:length(proposals)
        rv_m[proposal_index, :] = proposals[proposal_index].μ
        τ = 1.0
        _hess = hessian(x -> log(target(x)), rv_m[proposal_index, :])
        _test_lhs = target(rv_m[proposal_index, :] .+ τ .* _hess * d_log_target(rv_m[proposal_index, :]))
        _test_rhs = target(rv_m[proposal_index, :])
        # @info("LHS", _test_lhs)
        # @info("RHS", _test_rhs)
        while abs(_test_lhs / _test_rhs) >= 1.03
            τ /= 2.0
            # @info("τ", τ)
            _test_lhs = target(rv_m[proposal_index, :] .+ τ .* _hess * d_log_target(rv_m[proposal_index, :]))
            # @info("LHS", _test_lhs)
            if τ < 2^-20
                error("Convergence failure")
            end
        end
        @views rv_m[proposal_index, :] =
            rv_m[proposal_index, :] .+ τ .* _hess * d_log_target(rv_m[proposal_index, :])
    end
    return rv_m
end


function emscais!(
    target,
    samples,
    weights,
    transformed_weights;
    d_target = x -> gradient(k -> target(k), x)[1],
    d_log_target = x -> gradient(k -> log(target(k)), x)[1],
    proposals,
    iterations,
    samples_each,
    α = repeat([0.1], iterations),
    β = repeat([0.1], iterations),
    η = repeat([0.1], iterations),
    κ = repeat([0.1], iterations),
    γ = repeat([3.0], iterations),
    h = repeat([0.5], iterations),
    HMC_iters = iterations ÷ 2,
    N_t = round(Int, 1.5 * length(proposals[1])),
    HMC_args = [Dict() for x = 1:iterations],
)
    n_proposals = length(proposals)
    x_dim = length(proposals[begin])
    mdiff_arr = zeros(n_proposals, samples_each, x_dim)
    cov_arr = zeros(n_proposals, samples_each, x_dim, x_dim)
    cov_adapt = β .> 0.0
    log_target(x) = log(target(x))

    for i = 1:iterations
        HMC_args[i][:d_log_target] = d_log_target
    end

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
            cov_adapt = cov_adapt[iteration],
        )
        if iteration <= HMC_iters && κ[iteration] > 0
            @views μ2 = emscais_global_adapt!(log_target; proposals = proposals, HMC_args = HMC_args[iteration])
        else
            μ2 = μ1
        end
        # @views μ2 = emscais_gradient_adapt!(target, d_log_target; proposals = proposals, S = S[iteration])
        for proposal_index = 1:n_proposals
            @views μ =
                (1.0 .- α[iteration]) .* proposals[proposal_index].μ .+
                α[iteration] .* (1 - κ[iteration]) .* μ1[proposal_index, :] .+
                α[iteration] .* κ[iteration] .* μ2[proposal_index, :]

            proposals[proposal_index] = MvNormal(μ, Σ[proposal_index, :, :])
        end
        # if (iteration % 10 == 0)
        #     @info("EMSCAIS (DIM $x_dim): Iteration $iteration")
        # end
    end
    return (samples, weights)
end
