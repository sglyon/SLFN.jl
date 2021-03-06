"""

#### References

Enhanced random search based incremental extreme learning machine.

Guang-Bin Huang and Lei Chen.

Neurocomputing, 2008 vol. 71 (16-18) pp. 3460-3468.

http://linkinghub.elsevier.com/retrieve/pii/S0925231207003633
"""
mutable struct EIELM{TA,TV} <: AbstractSLFN where TA<:AbstractActivation where TV<:AbstractArray{Float64}
    p::Int              # Number of training points
    q::Int              # Dimensionality of function domain
    Lmax::Int           # maximum number of neurons
    k::Int              # maximum number of trials in node addition
    ϵ::Float64          # expected learning accuracy
    activation::TA
    μx::Vector{Float64}
    σx::Vector{Float64}
    At::Matrix{Float64} # transpose of W matrix
    b::Vector{Float64}
    v::TV

    function EIELM(p::Int, q::Int, Lmax::Int, k::Int, ϵ::Float64, activation::TA,
                   μx, σx) where TA
        WARNINGS[1] && @warn("This is experimental and doesn't work properly")
        At = Array{Float64}(undef, q, Lmax)  # will chop later
        b = Array{Float64}(undef, Lmax)      # ditto
        v = Array{Float64}(undef, Lmax)      # ditto
        new{TA,typeof(v)}(p, q, Lmax, k, ϵ, activation, μx, σx, At, b, v)
    end
end


function EIELM(x::AbstractArray, y::AbstractActivation;
               activation::AbstractActivation=Tanh(),
               Lmax::Int=size(x, 1), k::Int=20,
               ϵ::Float64=1e-6)
    q = size(x, 2)  # dimensionality of function domain
    p = size(x, 1)  # number of training points
    Lmax = min(Lmax, p)
    xn, μx, σx = normalize(x[:, :])
    out = EIELM(p, q, Lmax, k, ϵ, activation, μx, σx)
    fit!(out, xn, y)
end

## API methods
function fit!(elm::EIELM, x::AbstractArray, y::AbstractArray)
    # Step 1: Initialization
    L = 0
    E = copy(y)
    min_Ei = copy(E)

    # Step 2: learning
    while L < elm.Lmax && norm(E) > elm.ϵ
        L += 1
        min_Ei_contrib = Inf

        for i in 1:elm.k
            Ai = 2*rand(elm.q, 1) - 1  # uniform [-1, 1]
            bi = 2*rand() - 1          # uniform [-1, 1]
            Hi = hidden_out(elm, x, Ai, bi)
            vi = dot(E, Hi) / dot(Hi, Hi)
            Ei = E - vi*Hi
            Ei_contrib = norm(Ei)

            if Ei_contrib < min_Ei_contrib
                elm.At[:, L] = Ai
                elm.b[L] = bi
                elm.v[L] = vi
                min_Ei_contrib = Ei_contrib
                copy!(min_Ei, Ei)
            end

        end

        copy!(E, min_Ei)
    end

    # chop before returning
    elm.At = elm.At[:, 1:L]
    elm.b = elm.b[1:L]
    elm.v = elm.v[1:L]
    elm
end

function Base.show(io::IO, elm::EIELM{TA}) where TA
    s =
    """
    EIELM with
      - $(TA) Activation function
      - $(elm.q) input dimension(s)
      - $(length(elm.b)) neuron(s)
      - $(elm.p) training point(s)
    """
    print(io, s)
end
