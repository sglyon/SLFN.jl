"""

#### References

Extreme learning machine: Theory and applications.

Guang-Bin Huang, Qin-Yu Zhu, and Chee-Kheong Siew.

Neurocomputing, 2006 vol. 70 (1-3) pp. 489-501.

http://linkinghub.elsevier.com/retrieve/pii/S0925231206000385
"""
type ELM{TA<:AbstractActivation,TN<:AbstractNodeInput,TV<:AbstractArray{Float64}} <: AbstractSLFN
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    activation::TA
    neuron_type::TN
    Wt::Matrix{Float64}  # transpose of W matrix
    d::Vector{Float64}
    v::TV

    function ELM(p::Int, q::Int, s::Int, activation::TA, neuron_type::TN)
        Wt = 2*rand(q, s) - 1
        d = rand(s)
        new(p, q, s, activation, neuron_type, Wt, d)
    end
end

function ELM{TA<:AbstractActivation,
             TN<:AbstractNodeInput,
             TV<:AbstractArray}(x::AbstractArray, y::TV; activation::TA=SoftPlus(),
                                neuron_type::TN=Linear(), s::Int=size(x, 1))
    q = size(y, 2)  # dimensionality of function domain
    p = size(y, 1)  # number of training points
    s = min(p, s)   # Can't have more neurons than training points
    out = ELM{TA,TN,TV}(p, q, s, activation, neuron_type)
    fit!(out, x, y)
end

## API methods
function fit!(elm::ELM, x::AbstractArray, y::AbstractArray)
    S = hidden_out(elm, x)
    # elm.v = pinv(S) * u
    elm.v = S \ y
    elm
end

function (elm::ELM)(x′::AbstractArray)
    @assert size(x′, 2) == elm.q "wrong input dimension"
    return hidden_out(elm, x′) * elm.v
end

function Base.show{TA}(io::IO, elm::ELM{TA})
    s =
    """
    ELM with
      - $(TA) Activation function
      - $(elm.q) input dimension(s)
      - $(elm.s) neuron(s)
      - $(elm.p) training point(s)
    """
    print(io, s)
end
