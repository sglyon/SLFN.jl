abstract AbstractNodeInput

immutable Linear <: AbstractNodeInput end

# q > 1, s = 1; evaluated at one point
input_to_node(::Type{Linear}, x::AbstractVector, Wt::AbstractVector, d::Real) =
    dot(x, Wt) + d

# q > 1, s = 1; evaluated at many points
input_to_node(::Type{Linear}, x::AbstractMatrix, Wt::AbstractVector, d::Real) =
    x*Wt + d

# q = 1, s = 1; evaluated at one point
input_to_node(::Type{Linear}, x::Real, Wt::Real, d::Real) =
    x*Wt + d

# q = 1, s = 1; evaluated at many points
input_to_node(::Type{Linear}, x::AbstractVector, Wt::Real, d::Real) =
    x*Wt + d

# q > 1, s > 1;  evaluated at many points
input_to_node(::Type{Linear}, x::AbstractMatrix, Wt::AbstractMatrix, d::AbstractVector) =
    x*Wt .+ d'

# q > 1, s > 1;  evaluated at one point
input_to_node(::Type{Linear}, x::AbstractVector, Wt::AbstractMatrix, d::AbstractVector) =
    x*Wt .+ d'

input_to_node(::Linear, x, Wt, d) = input_to_node(Linear, x, Wt, d)

abstract AbstractRBFFamily

immutable Gaussian <: AbstractRBFFamily end
immutable RBF{Family<:AbstractRBFFamily} <: AbstractNodeInput end

RBF{TF<:AbstractRBFFamily}(::Union{TF,Type{TF}}) = RBF{TF}()

# q = 1, s = 1; evaluated at one point
input_to_node(::Type{RBF{Gaussian}}, x::AbstractVector, c::AbstractVector, σ::Real) =
    exp(-sqeuclidean(x, c)/σ)

# q > 1, s = 1; evaluated at many points
input_to_node(::Type{RBF{Gaussian}}, x::AbstractMatrix, c::AbstractVector, σ::Real) =
    exp(-colwise(SqEuclidean(), x', c) ./ σ)

# q = 1, s = 1; evaluated at one point
input_to_node(::Type{RBF{Gaussian}}, x::Real, c::Real, σ::Real) =
    exp(-(x-c)^2/σ)

# q = 1, s = 1; evaluated at many points
input_to_node(::Type{RBF{Gaussian}}, x::AbstractVector, c::Real, σ::Real) =
    exp(-(x-c).^2 ./ σ)

# q > 1, s > 1;  evaluated at many points
function input_to_node(::Type{RBF{Gaussian}}, x::AbstractMatrix, c::AbstractMatrix,
                       σ::AbstractVector)
    _out = Array(eltype(x), size(c, 2), size(x, 1))
    xp = x'
    for i in 1:size(x, 1)
        _out[:, i] = input_to_node(RBF{Gaussian}, view(xp, :, i), c, σ)
    end
    _out'
end

# q > 1, s > 1;  evaluated at one point
function input_to_node(::Type{RBF{Gaussian}}, x::AbstractVector, c::AbstractMatrix,
                       σ::AbstractVector)
    if size(c, 1) == 1
        # q = 1, s > 1; many points
        exp(-pairwise(SqEuclidean(), x', c) ./ σ')
    else
        exp(-colwise(SqEuclidean(), x, c') ./ σ)
    end
end

input_to_node(::RBF{Gaussian}, x, c, σ) = input_to_node(RBF{Gaussian}, x, c, σ)
