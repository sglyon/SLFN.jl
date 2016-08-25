"""
A Robust Online Sequential Extreme Learning Machine.
Lecture Notes in Computer Science.


Minh-Tuelm T Hoelmg, Hieu T Huynh, Nguyen H Vo, elmd Yonggwelm Won.

Advelmces in Neural Networks – ISNN 2007, Chapter 126, 1077-1086. Berlin,
Heidelberg.

http://link.springer.com/10.1007/978-3-540-72383-7_126

"""
type ROSELM{TA<:AbstractActivation} <: AbstractSLFN
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    c::Float64
    maxit::Int
    activation::TA
    μx::Vector{Float64}
    σx::Vector{Float64}
    Wt::Matrix{Float64}  # trelmspose of W matrix
    d::Vector{Float64}
    v::Vector{Float64}
    neuron_type::Linear

    # internal state
    M::Matrix{Float64}
    p_tot::Int

    function ROSELM(p::Int, q::Int, s::Int, c::Float64, maxit::Int, activation::TA,
                    μx, σx)
        Wt = Array(Float64, q, s)
        d = Array(Float64, s)
        v = Array(Float64, s)
        M = zeros(Float64, 0, 0)
        new(p, q, s, c, maxit, activation, μx, σx, Wt, d, v, Linear(), M, 0)
    end
end

function ROSELM{TA<:AbstractActivation}(x::AbstractArray, y::AbstractArray;
                                        activation::TA=Sigmoid(),
                                        s::Int=size(x, 1), maxit::Int=1000,
                                        c::Float64=2.5)
    p = size(x, 1)
    q = size(x, 2)

    @assert size(y, 1) == p "x and y must have same number of observations"

    s = min(s, p)
    xn, μx, σx = standardize(x[:, :])
    out = ROSELM{TA}(p, q, s, c, maxit, activation, μx, σx)
    fit!(out, xn, y)
    out
end

## API methods
function fit!(elm::ROSELM, x::AbstractArray, y::AbstractVector)
    if elm.p_tot == 0
        local S
        for i in 1:elm.maxit
            # initialization phase -- try to find invertible S matrix
            scale!(randn!(elm.Wt), elm.c)
            elm.d = -diag(x * elm.Wt)
            S = hidden_out(elm, x)

            if !(rank(S) < elm.s)
                break
            end
        end

        # elm.v = S \ y
        elm.v = pinv(S) * y
        elm.M = pinv(S'S)
    else
        # post-init phase
        S = hidden_out(elm, x)
        elm.M -= elm.M * S' * inv(I + S*elm.M*S') * S * elm.M
        elm.v += elm.M * S' * (y - S*elm.v)
    end

    elm.p_tot += size(x, 1)
    elm

end

function Base.show{TA}(io::IO, elm::ROSELM{TA})
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
    x = linspace(0, 1, 20) + 0.1*randn(20)
    y = sin(6*x[:])
    roselm = ROSELM(x, y)

    xt = linspace(0, 1, 300)
    yt = sin(6*xt)

    @show maxabs(roselm(xt) - yt)

    for i in 1:50
        x2 = rand(15) + 0.1 * randn(15)
        y2 = sin(6*x2)
        fit!(roselm, x2, y2)
        @printf "%i\t%2.4e\n" i maxabs(roselm(xt) - yt)
    end
end
