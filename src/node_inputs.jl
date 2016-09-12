abstract AbstractNodeInput

immutable Linear{TA<:AbstractActivation} <: AbstractNodeInput
    activation::TA
end

Linear{TA}(::Type{TA}) = Linear(TA())

# special case the event when both x and Wt are vectors
input_to_node(::Type{Linear}, x::AbstractVector, Wt::AbstractVector, d) =
    dot(x, Wt) + d'

# otherwise let dispatch work its magic
input_to_node{T<:Linear}(::Type{T}, x, Wt, d) = x*Wt .+ d'
input_to_node(::Linear, x, Wt, d) = input_to_node(Linear, x, Wt, d)
activate(l::Linear, h) = activate(l.activation, h)
hidden_out(l::Linear, x, Wt, d) = activate(l, input_to_node(l, x, Wt, d))

# Radial basis functions
abstract AbstractRBFFamily

immutable RBF{TF<:AbstractRBFFamily,TD<:Union{PreMetric,SemiMetric}} <: AbstractNodeInput
    dist::TD
end

function RBF{TF<:AbstractRBFFamily,
             TD<:Union{PreMetric,SemiMetric}}(::Union{TF,Type{TF}}, d::TD=SqEuclidean())
    RBF{TF,TD}(d)
end


# Gaussian RBF
immutable Gaussian <: AbstractRBFFamily end

input_to_node(r::RBF{Gaussian}, x::AbstractVector, c::AbstractVector, σ::Real) =
    exp(-evaluate(r.dist, x, c)/σ)

input_to_node(t::RBF{Gaussian}, x::AbstractMatrix, c::AbstractVector, σ::Real) =
    exp(-colwise(r.dist, x', c) ./ σ)

function input_to_node(r::RBF{Gaussian}, x::AbstractMatrix, c::AbstractMatrix,
                       σ::AbstractVector)
    _out = Array(eltype(x), size(c, 2), size(x, 1))
    xp = x'
    for i in 1:size(x, 1)
        _out[:, i] = input_to_node(r, view(xp, :, i), c, σ)
    end
    _out'
end

function input_to_node(r::RBF{Gaussian}, x::AbstractVector, c::AbstractMatrix,
                       σ::AbstractVector)
    if size(c, 1) == 1
        # q = 1, s > 1; many points
        exp(-pairwise(r.dist, x', c) ./ σ')
    else
        exp(-colwise(r.dist, x, c') ./ σ)
    end
end

# Special case scalar x version
input_to_node(r::RBF{Gaussian,SqEuclidean}, x::Real, c, σ) =
    exp(-(x-c).^2 ./ σ')

input_to_node(r::RBF{Gaussian,Euclidean}, x::Real, c, σ) =
    exp(-sqrt((x-c).^2) ./ σ')

hidden_out(rbf::RBF, x, c, σ) = input_to_node(rbf, x, c, σ)
