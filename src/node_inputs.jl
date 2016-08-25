abstract AbstractNodeInput

immutable Linear <: AbstractNodeInput end

# special case the event when both x and Wt are vectors
input_to_node(::Type{Linear}, x::AbstractVector, Wt::AbstractVector, d) =
    dot(x, Wt) + d'

# otherwise let dispatch work its magic
input_to_node(::Type{Linear}, x, Wt, d) = x*Wt .+ d'

input_to_node(::Linear, x, Wt, d) = input_to_node(Linear, x, Wt, d)

abstract AbstractRBFFamily

immutable Gaussian <: AbstractRBFFamily end
immutable RBF{Family<:AbstractRBFFamily} <: AbstractNodeInput end

RBF{TF<:AbstractRBFFamily}(::Union{TF,Type{TF}}) = RBF{TF}()

input_to_node(::Type{RBF{Gaussian}}, x::AbstractVector, c::AbstractVector, σ::Real) =
    exp(-sqeuclidean(x, c)/σ)

input_to_node(::Type{RBF{Gaussian}}, x::AbstractMatrix, c::AbstractVector, σ::Real) =
    exp(-colwise(SqEuclidean(), x', c) ./ σ)

function input_to_node(::Type{RBF{Gaussian}}, x::AbstractMatrix, c::AbstractMatrix,
                       σ::AbstractVector)
    _out = Array(eltype(x), size(c, 2), size(x, 1))
    xp = x'
    for i in 1:size(x, 1)
        _out[:, i] = input_to_node(RBF{Gaussian}, view(xp, :, i), c, σ)
    end
    _out'
end

function input_to_node(::Type{RBF{Gaussian}}, x::AbstractVector, c::AbstractMatrix,
                       σ::AbstractVector)
    if size(c, 1) == 1
        # q = 1, s > 1; many points
        exp(-pairwise(SqEuclidean(), x', c) ./ σ')
    else
        exp(-colwise(SqEuclidean(), x, c') ./ σ)
    end
end

# Special case scalar x version
input_to_node(::Type{RBF{Gaussian}}, x::Real, c, σ) =
    exp(-(x-c).^2 ./ σ')

input_to_node(::RBF{Gaussian}, x, c, σ) = input_to_node(RBF{Gaussian}, x, c, σ)
