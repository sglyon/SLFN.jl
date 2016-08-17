abstract AbstractNodeInput

immutable Linear <: AbstractNodeInput end

    
# q > 1, s = 1; evaluated at one point
input_to_node(::Type{Linear}, x::AbstractVector, Wt::AbstractVector, d::Real) = 
    dot(x, Wt) + d
    
# q > 1, s = 1; evaluated at many points
input_to_node(::Type{Linear}, x::AbstractVector, Wt::AbstractVector, d::Real) = 
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
    dot(x, Wt) .+ d'
    
abstract AbstractRBFFamily

immutable Gaussian <: AbstractRBFFamily end
immutable RBF{Family<:AbstractRBFFamily} <: AbstractNodeInput end