"""

#### References

Extreme learning machine: Theory and applications.

Guang-Bin Huang, Qin-Yu Zhu, and Chee-Kheong Siew.

Neurocomputing, 2006 vol. 70 (1-3) pp. 489-501.

http://linkinghub.elsevier.com/retrieve/pii/S0925231206000385
"""
mutable struct ELM{TN,TV} <: AbstractSLFN where TN <:AbstractNodeInput where TV <:AbstractArray{Float64}
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    neuron_type::TN
    μx::Vector{Float64}
    σx::Vector{Float64}
    Wt::Matrix{Float64}  # transpose of W matrix
    d::Vector{Float64}
    v::TV

    function ELM(p::Int, q::Int, s::Int, neuron_type::TN, μx, σx) where TN
        Wt = 2*rand(q, s) .- 1
        d = rand(s)
        new{TN,typeof(d)}(p, q, s, neuron_type, μx, σx, Wt, d)
    end
end

function ELM(x::AbstractArray, y::TV;
             neuron_type::TN=Linear(Tanh()), s::Int=size(x, 1),
             reg::AbstractLinReg=LSSVD()) where TN<:AbstractNodeInput where TV<:AbstractArray
    q = size(x, 2)  # dimensionality of function domain
    p = size(x, 1)  # number of training points
    s = min(p, s)   # Can't have more neurons than training points
    xn, μx, σx = standardize(x[:, :])
    # out = ELM{TN,TV}(p, q, s, neuron_type, μx, σx)
    out = ELM(p, q, s, neuron_type, μx, σx)
    fit!(out, xn, y, reg)
end

## API methods
function fit!(elm::ELM, x::AbstractArray, y::AbstractArray,
              reg::AbstractLinReg=LSSVD())
    S = hidden_out(elm, x)
    elm.v = regress(reg, S, y)
    elm
end

function Base.show(io::IO, elm::ELM{TA}) where TA
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
