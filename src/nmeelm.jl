"""
Nelder-mead enhanced extreme learning machine.

Philip Reiner and Bogdan M Wilamowski.

2013 IEEE 17th International Conference on Intelligent Engineering Systems
(INES), 2013 pp. 225-230.

http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=6632816

"""
type NMEELM{TA<:AbstractActivation,TN<:AbstractNodeInput,TV<:AbstractArray{Float64}} <: AbstractSLFN
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    activation::TA
    neuron_type::TN
    ρ::Float64
    χ::Float64
    γ::Float64
    α::Float64
    Wt::Matrix{Float64}  # transpose of W matrix
    d::Vector{Float64}
    v::TV

    function NMEELM(p::Int, q::Int, s::Int, activation::TA, neuron_type::TN,
                    ρ::Float64, χ::Float64, γ::Float64, α::Float64)
        Wt = 2*rand(q, s) - 1
        d = rand(s)
        new(p, q, s, activation, neuron_type, ρ, χ, γ, α, Wt, d)
    end
end

function NMEELM{TA<:AbstractActivation,
             TN<:AbstractNodeInput,
             TV<:AbstractArray}(y::AbstractArray, u::TV, activation::TA=SoftPlus(),
                                neuron_type::TN=Linear(), s::Int=size(y, 1))
    q = size(y, 2)  # dimensionality of function domain
    p = size(y, 1)  # number of training points
    s = min(p, s)   # Can't have more neurons than training points
    out = NMEELM{TA,TN,TV}(p, q, s, activation, neuron_type)
    fit!(out, y, u)
end

## API methods
isexact(elm::NMEELM) = false
input_to_node(elm::NMEELM, y::AbstractArray) = input_to_node(elm.neuron_type, y, elm.Wt, elm.d)
hidden_out(elm::NMEELM, y::AbstractArray) = elm.activation(input_to_node(elm, y))

function fit!(elm::NMEELM, y::AbstractArray, u::AbstractArray)
    S = hidden_out(elm, y)
    @show size(S)
    elm.v = S \ u
    elm
end

function (elm::NMEELM)(y′::AbstractArray)
    @assert size(y′, 2) == elm.q "wrong input dimension"
    return hidden_out(elm, y′) * elm.v
end

function Base.show{TA}(io::IO, elm::NMEELM{TA})
    s =
    """
    NMEELM with
      - $(TA) Activation function
      - $(elm.q) input dimension(s)
      - $(elm.s) neuron(s)
      - $(elm.p) training point(s)
    """
    print(io, s)
end
