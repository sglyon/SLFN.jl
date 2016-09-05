"""

#### References

An incremental extreme learning machine for online sequential learning problems.

Lu Guo, Jing-hua Hao, and Min Liu.

Neurocomputing, 2014 vol. 128 pp. 50-58.

http://linkinghub.elsevier.com/retrieve/pii/S0925231213010059

"""
type LSIELM{TN<:AbstractNodeInput,TV<:AbstractArray{Float64}} <: AbstractSLFN
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    neuron_type::TN
    c::Float64  # regularization parameter
    μx::Vector{Float64}
    σx::Vector{Float64}
    Wt::Matrix{Float64}  # transpose of W matrix
    d::Vector{Float64}

    M::Matrix{Float64}  # internal
    p_tot::Int
    v::TV

    function LSIELM(p::Int, q::Int, s::Int, neuron_type::TN, c::Float64,
                    μx, σx)
        # NOTE: Wt and d are constant throughout all learning chunks
        Wt = 2*rand(q, s) - 1
        d = rand(s)
        M = zeros(Float64, s, s)
        new(p, q, s, neuron_type, c, μx, σx, Wt, d, M, 0)
    end
end

function LSIELM{TN<:AbstractNodeInput,
                TV<:AbstractArray}(x::AbstractArray, y::TV;
                                   c::Float64=1/(50*eps()),
                                   neuron_type::TN=Linear(Tanh()),
                                   s::Int=size(x, 1))
    q = size(x, 2)  # dimensionality of function domain
    p = size(x, 1)  # number of training points
    s = min(p, s)   # Can't have more neurons than training points
    xn, μx, σx = standardize(x[:, :])
    out = LSIELM{TN,TV}(p, q, s, neuron_type, c, μx, σx)
    fit!(out, xn, y)
end

## API methods
function fit!(elm::LSIELM, x::AbstractArray, y::AbstractArray)
    # check if we are boosting or not
    S = hidden_out(elm, x)
    if elm.p_tot == 0  # this is the initialization phase
        # elm.v = S \ y
        elm.M = inv(1/elm.c + S'S)
        elm.v = elm.M*S'y
    else
        K = I - elm.M * S' * inv(S*elm.M*S' + I)*S
        elm.v = K*(elm.v + elm.M*S'y)
        elm.M = K*elm.M
    end

    elm.p_tot += size(x, 1)
    elm

end

function Base.show{TA}(io::IO, elm::LSIELM{TA})
    s =
    """
    LSIELM with
      - $(TA) Activation function
      - $(elm.q) input dimension(s)
      - $(elm.s) neuron(s)
      - $(elm.p) initial training point(s)
      - $(elm.p_tot) total training points
    """
    print(io, s)
end


function test_me(::Type{LSIELM})
    x = linspace(0, 1, 20)
    y = sin(6*x[:])
    oselm = LSIELM(x, y)

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
