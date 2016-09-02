abstract AbstractActivation

immutable Sigmoid <: AbstractActivation end
immutable SoftPlus <: AbstractActivation end
immutable Relu <: AbstractActivation end
immutable Tanh <: AbstractActivation end
immutable Identity <: AbstractActivation end

activate(::Sigmoid, x::Number) = 1 ./ (1 + exp(-x))
activate(::SoftPlus, x::Number) = log(1 + exp(x))
activate(::Relu, x::Number) = max(x, 0)
activate(::Tanh, x::Number) = tanh(x)
activate(::Identity, x::Number) = x

activate(a::AbstractActivation, x::AbstractArray) =
    map(_->activate(a, _), x)

activate!(out::AbstractArray, a::AbstractActivation, x::AbstractArray) =
    map!(_->activate(a, _), out, x)


for T in subtypes(AbstractActivation)
    @eval @compat (ta::$(T))(x) = activate(ta, x)
end

deriv(::Sigmoid, x) = exp(-x) ./((1 + exp(-x)).^2)
deriv(::SoftPlus, x) = exp(x) ./ (1 + exp(x))
deriv(::Relu, x::Number) = x >= 0.0 ? 1.0 : 0.0
deriv(r::Relu, x::AbstractArray) = map(_->deriv(r, _), x)
deriv(::Tanh, x) = 1 - tanh(x).^2
deriv(::Identity, x::Number) = one(x)
deriv(::Identity, x::AbstractArray) = ones(x)
