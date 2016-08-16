abstract AbstractActivation

immutable Sigmoid <: AbstractActivation end
immutable SoftPlus <: AbstractActivation end
immutable Relu <: AbstractActivation end
immutable Tanh <: AbstractActivation end

activate(::Sigmoid, x) = 1 ./ (1 + exp(-x))
activate(::SoftPlus, x) = log(1 + exp(x))
activate(::Relu, x) = max(x, 0)
activate(::Tanh, x) = tanh(x)

for T in subtypes(AbstractActivation)
    @eval (ta::$(T))(x) = activate(ta, x)
end

deriv(::Sigmoid, x) = exp(-x) ./((1 + exp(-x)).^2)
deriv(::SoftPlus, x) = exp(x) ./ (1 + exp(x))
deriv(::Relu, x::Number) = x >= 0.0 ? 1.0 : 0.0
deriv(r::Relu, x::AbstractArray) = map(_->deriv(r, _), x)
deriv(::Tanh, x) = 1 - tanh(x).^2
