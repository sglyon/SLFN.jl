abstract type AbstractActivation end

struct Sigmoid <: AbstractActivation end
struct SoftPlus <: AbstractActivation end
struct Relu <: AbstractActivation end
struct Tanh <: AbstractActivation end
struct Identity <: AbstractActivation end

activate(::Sigmoid, x::Number) = 1 ./ (1 + exp(-x))
activate(::SoftPlus, x::Number) = log(1 + exp(x))
activate(::Relu, x::Number) = max(x, 0)
activate(::Tanh, x::Number) = tanh(x)
activate(::Identity, x::Number) = x

activate(a::AbstractActivation, x::AbstractArray) =
    map(_1->activate(a, _1), x)

activate!(out::AbstractArray, a::AbstractActivation, x::AbstractArray) =
    map!(_1->activate(a, _1), out, x)


for T in InteractiveUtils.subtypes(AbstractActivation)
    @eval (ta::$(T))(x) where T = activate(ta, x)
end

deriv(::Sigmoid, x) = exp(-x) ./((1 + exp(-x)).^2)
deriv(::SoftPlus, x) = exp(x) ./ (1 + exp(x))
deriv(::Relu, x::Number) = x >= 0.0 ? 1.0 : 0.0
deriv(r::Relu, x::AbstractArray) = map(_1->deriv(r, _1), x)
deriv(::Tanh, x) = 1 - tanh(x).^2
deriv(::Identity, x::Number) = one(x)
deriv(::Identity, x::AbstractArray) = ones(x)
