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

@inline target(x) = banana(x)
@inline d_ltarget(x) = g_l_banana(x)

n_props = 20
n_iters = 150
n_spi = 200
prop_sigma = 1.0
x_dim = 30

samples = zeros(n_iters, n_props, n_spi, x_dim)
s_weights = zeros(n_iters, n_props, n_spi)
t_weights = zeros(n_iters, n_props, n_spi)

props = init_proposals(x_dim, n_props, σ = prop_sigma, _rand_ = true)

β = [0.05 + (n_iters-i) * (0.3 / n_iters) for i in 1:n_iters]
α = [(0.1 + (n_iters-i) * (0.8 / n_iters))^(1.5) for i in 1:n_iters]
η = [inv(i) for i in 1:n_iters]
κ = [inv(i) for i in 1:n_iters]
HMC_args = [Dict(:d_log_target => g_l_banana) for x = 1:n_iters]

emscais!(target, samples, s_weights, t_weights, proposals = props, iterations = n_iters, samples_each = n_spi, β = β, α = α, η = η, κ = κ)

Zhat = mean(s_weights)
ism = inv(Zhat) * mean(s_weights .* samples, dims = [1,2,3])[1,1,1,:]
