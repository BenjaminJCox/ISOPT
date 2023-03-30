using Distributions
using StatsBase
using DrWatson
using CairoMakie
using Random
using KernelDensity
using Zygote
using JLD2
using BenchmarkTools

include(srcdir("emscais.jl"))

@inline function banana(x; b = 3.0, σ = 1.0)
    @assert length(x) > 2
    t1 = -x[1]^2 / (2σ^2)
    t2 = -((x[2] + b*(x[1]^2-σ^2))^2) / (2σ^2)
    t3 = -sum((x[3:end].^2) ./ (2σ^2))
    return exp(t1 + t2 + t3)
end

@inline function g_l_banana(x; b = 3.0, σ = 1.0)
    @assert length(x) > 2
    _rv = zero(x)
    _rv[3:end] = -x[3:end] ./ (σ^2)
    _rv[2] = - (b * (x[1]^2 - σ^2) + x[2])/(σ^2)
    _rv[1] = -(x[1] ./ (σ^2)) -(-2*b^2*σ^2*x[1] + 2*b^2*x[1]^3 + 2*b*x[1]*x[2])/(σ^2)
    return _rv
end

@inline function g_banana(x; b = 3.0, σ = 1.0)
    @assert length(x) > 2
    efx = banana(x; b = b, σ = σ)
    fpx = g_l_banana(x, b = b, σ = σ)
    return efx .* fpx
end


@inline target(x) = banana(x)
@inline d_ltarget(x) = g_l_banana(x)
@inline d_target(x) = g_banana(x)

n_props = 20
n_iters = 150
n_spi = 200
prop_sigma = 1.0
# x_dim = 25

# samples = zeros(n_iters, n_props, n_spi, x_dim)
# s_weights = zeros(n_iters, n_props, n_spi)
# t_weights = zeros(n_iters, n_props, n_spi)

β = [0.05 + (n_iters-i) * (0.5 / n_iters) for i in 1:n_iters]
β[1:5] .= 0.0
α = [(0.01 + (n_iters-i) * (0.99 / n_iters)) for i in 1:n_iters].^2
η = [inv(i) for i in 1:n_iters]
κ = [(0.1 + (n_iters-i) * (0.2 / n_iters)) for i in 1:n_iters]
# κ .= 0.0
ϵ = [(0.02 + (n_iters-i) * (0.08 / n_iters)) for i in 1:n_iters]
L = Int.(round.(collect(range(3, 1, length = n_iters))))
hmciters = Int.(round.(collect(range(2, 1, length = n_iters))))
K = [(0.02 + (n_iters-i) * (0.98 / n_iters)) for i in 1:n_iters]
K[end-50:end] .= 0.0
HMC_args = [Dict(:ϵ => ϵ[i], :L => L[i], :d_log_target => d_ltarget, :repulsion => i < 100, :K => K[i], :n_iterations => hmciters[i]) for i = 1:n_iters]
use_iters = n_iters ÷ 3

# props = init_proposals(10, n_props, σ = prop_sigma, _rand_ = false, lims = (-2, 2))

dims = [3, 5, 7, 10, 15, 20, 25, 30]
# dims = [20]
MSE = zeros(size(dims))

props = init_proposals(10, n_props, σ = prop_sigma, _rand_ = false, lims = (-2, 2))

for _idx in 1:length(dims)
    x_dim = dims[_idx]

    samples = zeros(n_iters, n_props, n_spi, x_dim)
    s_weights = zeros(n_iters, n_props, n_spi)
    t_weights = zeros(n_iters, n_props, n_spi)

    props .= init_proposals(x_dim, n_props, σ = prop_sigma, _rand_ = false, lims = (-2, 2))

    emscais!(target, samples, s_weights, t_weights, d_target = g_banana, d_log_target = g_l_banana, proposals = props, iterations = n_iters, samples_each = n_spi, β = β, α = α, η = η, κ = κ, HMC_args = HMC_args, HMC_iters = 50)

    @views e_samples = samples[use_iters:end, :, :, :]
    @views e_weights = s_weights[use_iters:end, :, :]

    Zhat = mean(e_weights)
    ism = inv(Zhat) * mean(e_weights .* e_samples, dims = [1,2,3])[1,1,1,:]
    mse = mean(ism .^ 2)
    MSE[_idx] = mse
    @info("EMSCAIS (DIM $x_dim) COMPLETE")
end

# Zhat = mean(s_weights)
# ism = inv(Zhat) * mean(s_weights .* samples, dims = [1,2,3])[1,1,1,:]
# mse = mean(ism .^ 2)
# display(mse)
display(MSE)
display(mean(MSE))
