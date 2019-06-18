"""
A Robust Online Sequential Extreme Learning Machine.
Lecture Notes in Computer Science.


Minh-Tuelm T Hoelmg, Hieu T Huynh, Nguyen H Vo, elmd Yonggwelm Won.

Advelmces in Neural Networks – ISNN 2007, Chapter 126, 1077-1086. Berlin,
Heidelberg.

http://link.springer.com/10.1007/978-3-540-72383-7_126

"""
mutable struct ROSELM{TN} <: AbstractSLFN where TN <: Linear
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    c::Float64
    maxit::Int
    μx::Vector{Float64}
    σx::Vector{Float64}
    Wt::Matrix{Float64}  # trelmspose of W matrix
    d::Vector{Float64}
    v::Vector{Float64}
    neuron_type::TN

    # internal state
    M::Matrix{Float64}
    p_tot::Int

    function ROSELM(p::Int, q::Int, s::Int, c::Float64, maxit::Int,
                    neuron_type::TN, μx, σx) where TN
        Wt = Array{Float64}(undef, q, s)
        d = Array{Float64}(undef, s)
        v = Array{Float64}(undef, s)
        M = zeros(Float64, 0, 0)
        new{TN}(p, q, s, c, maxit, μx, σx, Wt, d, v, neuron_type, M, 0)
    end
end

function ROSELM(x::AbstractArray, y::AbstractVector;
                neuron_type::TN=Linear(Tanh()),
                s::Int=size(x, 1), maxit::Int=1000,
                c::Float64=2.5) where TN <: Linear
    p = size(x, 1)
    q = size(x, 2)

    @assert size(y, 1) == p "x and y must have same number of observations"

    s = min(s, p)
    xn, μx, σx = standardize(x[:, :])
    out = ROSELM(p, q, s, c, maxit, neuron_type, μx, σx)
    fit!(out, xn, y)
    out
end

## API methods
function fit!(elm::ROSELM, x::AbstractArray, y::AbstractVector)
    if elm.p_tot == 0
        local S
        for i in 1:elm.maxit
            # initialization phase -- try to find invertible S matrix
            randn!(elm.Wt)
            elm.Wt ./= elm.c
            elm.d = -diag(x * elm.Wt)
            S = hidden_out(elm, x)

            if !(rank(S) < elm.s)
                break
            end
        end

        # elm.v = S \ y
        elm.M = pinv(S'S)
        elm.v = elm.M*S'*y
    else
        # post-init phase
        S = hidden_out(elm, x)
        elm.M -= elm.M * S' * inv(I + S*elm.M*S') * S * elm.M
        elm.v += elm.M * S' * (y - S*elm.v)
    end

    elm.p_tot += size(x, 1)
    elm

end

function Base.show(io::IO, elm::ROSELM{TA}) where TA
    s =
    """
    ROSELM with
      - $(TA) Activation function
      - $(elm.q) input dimension(s)
      - $(elm.s) neuron(s)
      - $(elm.p) training point(s)
    """
    print(io, s)
end

function test_me(::Type{ROSELM})
    x = range(0, stop=1, length=20) + 0.1*randn(20)
    y = sin.(6*x[:])
    roselm = ROSELM(x, y)

    xt = range(0, stop=1, length=300)
    yt = sin.(6*xt)

    @show maximum(abs.(roselm(xt) - yt))

    for i in 1:50
        x2 = rand(15) + 0.1 * randn(15)
        y2 = sin.(6*x2)
        fit!(roselm, x2, y2)
        @printf "%i\t%2.4e\n" i maximum(abs.(roselm(xt) - yt))
    end
end
