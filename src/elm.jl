type ELM{TA<:AbstractActivation,TV<:AbstractArray{Float64}} <: AbstractSLFN
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    activation::TA
    Wt::Matrix{Float64}  # transpose of W matrix
    d::Vector{Float64}
    v::TV

    function ELM(p::Int, q::Int, s::Int, activation::TA)
        Wt = 2*rand(q, s) - 1
        d = rand(s)
        new(p, q, s, activation, Wt, d)
    end
end


function ELM{TA<:AbstractActivation,TV<:AbstractArray}(y::AbstractArray, u::TV,
                                                       activation::TA=SoftPlus(),
                                                       s::Int=size(y, 1))
    q = size(y, 2)  # dimensionality of function domain
    p = size(y, 1)  # number of training points
    s = min(p, s)   # Can't have more neurons than training points
    out = ELM{TA,TV}(p, q, s, activation)
    fit!(out, y, u)
end

## API methods
isexact(elm::ELM) = elm.p == elm.s
input_to_node(elm::ELM, y::AbstractArray) = y*elm.Wt .+ elm.d'
hidden_out(elm::ELM, y::AbstractArray) = elm.activation(input_to_node(elm, y))

function fit!(elm::ELM, y::AbstractArray, u::AbstractArray)
    S = hidden_out(elm, y)
    elm.v = S \ u
    elm
end

function (elm::ELM)(y′::AbstractArray)
    @assert size(y′, 2) == elm.q "wrong input dimension"
    return hidden_out(elm, y′) * elm.v
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