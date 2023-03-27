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

@inline target(x) = banana(x)

n_props = 20
n_iters = 5
n_spi = 200
prop_sigma = 2.0
x_dim = 15

samples = zeros(n_iters, n_props, n_spi, x_dim)
s_weights = zeros(n_iters, n_props, n_spi)
t_weights = zeros(n_iters, n_props, n_spi)

props = init_proposals(x_dim, n_props, σ = prop_sigma, _rand_ = true)

β = [0.05 + (n_iters-i) * (0.3 / n_iters) for i in 1:n_iters]
α = [0.05 + (n_iters-i) * (0.6 / n_iters) for i in 1:n_iters]
η = [inv(i) for i in 1:n_iters]

@btime emscais!(target, samples, s_weights, t_weights, proposals = props, iterations = n_iters, samples_each = n_spi, β = β, α = α, η = η)

Zhat = mean(s_weights)
ism = inv(Zhat) * mean(s_weights .* samples, dims = [1,2,3])[1,1,1,:]
