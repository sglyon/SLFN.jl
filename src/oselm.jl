"""

#### References

A Fast and Accurate Online Sequential Learning Algorithm for Feedforward
Networks

Guang-Bin Huang, Qin-Yu Zhu, and Chee-Kheong Siew.

Neurocomputing, 2006 vol. 70 (1-3) pp. 489-501.

http://linkinghub.elsevier.com/retrieve/pii/S0925231206000385
"""
mutable struct OSELM{TN,TV} <: AbstractSLFN where TN <: AbstractNodeInput where TV <: AbstractArray{Float64}
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    neuron_type::TN
    μx::Vector{Float64}
    σx::Vector{Float64}
    Wt::Matrix{Float64}  # transpose of W matrix
    d::Vector{Float64}

    M::Matrix{Float64}  # internal
    p_tot::Int
    v::TV

    function OSELM(p::Int, q::Int, s::Int, neuron_type::TN, μx, σx) where TN
        # NOTE: Wt and d are constant throughout all learning chunks
        Wt = 2*rand(q, s) .- 1
        d = rand(s)
        M = zeros(Float64, s, s)
        new{TN,typeof(d)}(p, q, s, neuron_type, μx, σx, Wt, d, M, 0)
    end
end

function OSELM(x::AbstractArray, y::TV;
               neuron_type::TN=Linear(Tanh()),
               s::Int=size(x, 1)) where TN <: AbstractNodeInput where TV <: AbstractArray
    q = size(x, 2)  # dimensionality of function domain
    p = size(x, 1)  # number of training points
    s = min(p, s)   # Can't have more neurons than training points
    xn, μx, σx = standardize(x[:, :])
    out = OSELM(p, q, s, neuron_type, μx, σx)
    fit!(out, xn, y)
end

## API methods
function fit!(elm::OSELM, x::AbstractArray, y::AbstractArray)
    # check if we are boosting or not
    S = hidden_out(elm, x)
    if elm.p_tot == 0  # this is the initialization phase
        # elm.v = S \ y
        elm.M = pinv(S'S)
        elm.v = elm.M*S'y
    else
        elm.M -= elm.M*S' * inv(I + S*elm.M*S') * S * elm.M
        elm.v += elm.M * S' * (y - S*elm.v)
    end
    elm.p_tot += size(x, 1)
    elm
end

function Base.show(io::IO, elm::OSELM{TA}) where TA
    s =
    """
    OSELM with
      - $(TA) Activation function
      - $(elm.q) input dimension(s)
      - $(elm.s) neuron(s)
      - $(elm.p) initial training point(s)
      - $(elm.p_tot) total training points
    """
    print(io, s)
end


function test_me(::Type{OSELM})
    x = range(0, stop=1, length=20)
    y = sin.(6*x[:])
    oselm = OSELM(x, y)

    xt = range(0, stop=1, length=300)
    yt = sin.(6*xt)

    @show maximum(abs.(oselm(xt) .- yt))

    for i in 1:50
        x2 = rand(15)
        y2 = sin.(6*x2)
        fit!(oselm, x2, y2)
        @printf "%i\t%2.4e\n" i maximum(abs.(oselm(xt) - yt))
    end
end
