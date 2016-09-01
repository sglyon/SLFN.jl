"""

#### References

Smooth function approximation using neural networks.

S Ferrari and R F Stengel.

IEEE Trans Neural Netw, 2005 vol. 16 (1) pp. 24-38.

http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=1388456
"""
type AlgebraicNetwork{TN<:Linear} <: AbstractSLFN
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    n_train_it::Int  # number of training iterations
    f::Float64
    maxit::Int
    neuron_type::TN
    μx::Vector{Float64}
    σx::Vector{Float64}
    Wt::Matrix{Float64}  # transpose of W matrix
    d::Vector{Float64}
    v::Vector{Float64}

    function AlgebraicNetwork(p::Int, q::Int, s::Int, neuron_type::TN,
                              f::Float64, maxit::Int, μx, σx)
        Wt = Array(Float64, q, s)
        d = Array(Float64, s)
        v = Array(Float64, s)
        new(p, q, s, 0, f, maxit, neuron_type, μx, σx, Wt, d, v)
    end
end

function AlgebraicNetwork{TN<:Linear}(x::AbstractArray, y::AbstractArray;
                                      neuron_type::TN=Linear(Tanh()),
                                      s::Int=size(x, 1), f::Float64=0.8,
                                      maxit::Int=1000,
                                      reg::AbstractLinReg=LSSVD())
    p = size(x, 1)
    q = size(x, 2)

    @assert size(y, 1) == p "x and y must have same number of observations"

    s = min(s, p)
    xn, μx, σx = standardize(x[:, :])
    out = AlgebraicNetwork{TN}(p, q, s, neuron_type, f, maxit, μx, σx)
    fit!(out, xn, y, reg)
    out
end

## API methods
isexact(an::AlgebraicNetwork) = an.p == an.s
function fit!(an::AlgebraicNetwork, x::AbstractArray, y::AbstractVector,
              reg::AbstractLinReg)
    i = 0
    while true
        i += 1
        scale!(randn!(an.Wt), an.f)
        # fill!(an.Wt, 5.0)
        an.d = -diag(x * an.Wt)
        S = hidden_out(an, x)

        if rank(S) < an.s && i < an.maxit
            an.n_train_it += 1
            continue
        else
            an.v = StableReg.regress(reg, S, y)
            return an
        end
    end

end

function Base.show{TA}(io::IO, an::AlgebraicNetwork{TA})
    s =
    """
    AlgebraicNetwork with
      - $(TA) Activation function
      - $(an.q) input dimension(s)
      - $(an.s) neuron(s)
      - $(an.p) training point(s)
      - $(an.n_train_it) training iterations
    """
    print(io, s)
end
