"""
Nelder-mead enhanced extreme learning machine.

Philip Reiner and Bogdan M Wilamowski.

2013 IEEE 17th International Conference on Intelligent Engineering Systems
(INES), 2013 pp. 225-230.

http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=6632816

"""
mutable struct NMEELM{TV<:AbstractArray{Float64}} <: AbstractSLFN
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    ϵ::Float64  # expected learning accuracy

    # Nelder Mead parameters
    ρ::Float64
    χ::Float64
    γ::Float64
    α::Float64

    # Number of simplex operations
    k::Int

    # internal parameters
    Wt::Matrix{Float64}  # transpose of W matrix
    d::Vector{Float64}
    activation::Identity
    neuron_type::RBF{Gaussian}
    β::TV

    function NMEELM(p::Int, q::Int, s::Int, ϵ::Float64,
                    ρ::Float64, χ::Float64, γ::Float64, α::Float64)
        Wt = 2*rand(q, s) - 1
        d = rand(s)
        β = TV(s)
        new(p, q, s, ϵ, ρ, χ, γ, α, Wt, d, Identity(), RBF(Gaussian), β)
    end
end

function NMEELM{TV<:AbstractArray}(x::AbstractArray, y::TV; s::Int=size(y, 1),
                                   ϵ::Float64=1e-6, ρ::Float64=1.0, χ::Float64=2.0,
                                   γ::Float64=1/2, α::Float64=1/2, k::Int=7)
    q = size(x, 2)  # dimensionality of function domain
    p = size(y, 1)  # number of training points
    s = min(p, s)   # Can't have more neurons than training points
    out = NMEELM{TV}(p, q, s, ϵ, ρ, χ, γ, α, k)
    fit!(out, x, y)
end

## API methods
isexact(elm::NMEELM) = false
input_to_node(elm::NMEELM, x::AbstractArray) = input_to_node(elm.neuron_type, x, elm.Wt, elm.d)
hidden_out(elm::NMEELM, x::AbstractArray) = elm.activation(input_to_node(elm, x))


function fit!(elm::NMEELM, x::AbstractArray, y::AbstractArray)
    # ------- Step 1: initialize
    n = 0
    E = copy(y)
    err = Inf
    xt = x'

    # ------- Step 2: Learning
    while n < elm.s && err > elm.ϵ
        # --- Step 2.a: Increase n
        n += 1

        # --- Step 2.b: find index of max error
        j = indmax(E)

        # --- Step 2.c: Assign new center to be input x
        cn = xt[:, j]
        elm.Wt[:, n] = cn

        # --- Step 2.d: Initialze βₙ = Eⱼ
        βn = E[j]

        # --- Step 2.e: Do Nelder-Mead simplex algorithm
        # define objective function
        function obj(σ)
            sse = 0.0
            for i in 1:elm.p
                gi = input_to_node(elm.neuron_type, view(xt, :, i), cn, σ)
                sse += (E[i] - βn * gi)^2
            end
            sse
        end

        # call to elm.k nm simplex iterations (or call Optim??)
        # TODO: pick up here


    end

end

@compat function (elm::NMEELM)(x′::AbstractArray)
    @assert size(x′, 2) == elm.q "wrong input dimension"
    return hidden_out(elm, x′) * elm.v
end

function Base.show{TA}(io::IO, elm::NMEELM{TA})
    s =
    """
    NMEELM with
      - Identity Activation function
      - $(elm.q) input dimension(s)
      - RBF{Gaussian} neuron(s)
      - $(elm.p) training point(s)
    """
    print(io, s)
end
