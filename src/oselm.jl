"""

#### References

A Fast and Accurate Online Sequential Learning Algorithm for Feedforward
Networks

Guang-Bin Huang, Qin-Yu Zhu, and Chee-Kheong Siew.

Neurocomputing, 2006 vol. 70 (1-3) pp. 489-501.

http://linkinghub.elsevier.com/retrieve/pii/S0925231206000385
"""
type OSELM{TA<:AbstractActivation,TN<:AbstractNodeInput,TV<:AbstractArray{Float64}} <: AbstractSLFN
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    activation::TA
    neuron_type::TN
    Wt::Matrix{Float64}  # transpose of W matrix
    d::Vector{Float64}

    M::Matrix{Float64}  # internal
    p_tot::Int
    v::TV

    function OSELM(p::Int, q::Int, s::Int, activation::TA, neuron_type::TN)
        # NOTE: Wt and d are constant throughout all learning chunks
        Wt = 2*rand(q, s) - 1
        d = rand(s)
        M = zeros(Float64, s, s)
        new(p, q, s, activation, neuron_type, Wt, d, M, 0)
    end
end

function OSELM{TA<:AbstractActivation,
               TN<:AbstractNodeInput,
               TV<:AbstractArray}(x::AbstractArray, y::TV; activation::TA=SoftPlus(),
                                  neuron_type::TN=Linear(), s::Int=size(x, 1))
    q = size(x, 2)  # dimensionality of function domain
    p = size(x, 1)  # number of training points
    s = min(p, s)   # Can't have more neurons than training points
    out = OSELM{TA,TN,TV}(p, q, s, activation, neuron_type)
    fit!(out, x, y)
end

## API methods
function fit!(elm::OSELM, x::AbstractArray, y::AbstractArray)
    # check if we are boosting or not
    S = hidden_out(elm, x)
    if elm.p_tot == 0  # this is the initialization phase
        # elm.v = S \ y
        elm.v = pinv(S) * y
        elm.M = pinv(S'S)
    else
        elm.M -= elm.M*S' * inv(I + S*elm.M*S') * S * elm.M
        elm.v += elm.M * S' * (y - S*elm.v)
    end

    elm.p_tot += size(x, 1)
    elm

end

@compat function (elm::OSELM)(x′::AbstractArray)
    @assert size(x′, 2) == elm.q "wrong input dimension"
    return hidden_out(elm, x′) * elm.v
end

function Base.show{TA}(io::IO, elm::OSELM{TA})
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
    x = linspace(0, 1, 20)
    y = sin(6*x[:])
    oselm = OSELM(x, y)

    xt = linspace(0, 1, 300)
    yt = sin(6*xt)

    @show maxabs(oselm(xt) - yt)

    for i in 1:50
        x2 = rand(15)
        y2 = sin(6*x2)
        fit!(oselm, x2, y2)
        @printf "%i\t%2.4e\n" i maxabs(oselm(xt) - yt)
    end
end
