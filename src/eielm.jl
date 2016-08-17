"""

#### References

Enhanced random search based incremental extreme learning machine.

Guang-Bin Huang and Lei Chen.

Neurocomputing, 2008 vol. 71 (16-18) pp. 3460-3468.

http://linkinghub.elsevier.com/retrieve/pii/S0925231207003633
"""
type EIELM{TA<:AbstractActivation,TV<:AbstractArray{Float64}} <: AbstractSLFN
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    Lmax::Int  # maximum number of neurons
    k::Int  # maximum number of trials in node addition
    ϵ::Float64  # expected learning accuracy
    activation::TA
    At::Matrix{Float64}  # transpose of W matrix
    b::Vector{Float64}
    β::TV

    function EIELM(p::Int, q::Int, Lmax::Int, k::Int, ϵ::Float64, activation::TA)
        At = Array(Float64, q, Lmax)  # will chop later
        b = Array(Float64, Lmax)      # ditto
        β = Array(Float64, Lmax)      # ditto
        new(p, q, Lmax, k, ϵ, activation, At, b, β)
    end
end


function EIELM{TA<:AbstractActivation,TV<:AbstractArray}(x::AbstractArray, t::TV,
                                                         activation::TA=SoftPlus(),
                                                         Lmax::Int=size(x, 1), k::Int=20,
                                                         ϵ::Float64=1e-6)
    q = size(x, 2)  # dimensionality of function domain
    p = size(x, 1)  # number of training points
    Lmax = min(Lmax, p)
    out = EIELM{TA,TV}(p, q, Lmax, k, ϵ, activation)
    fit!(out, x, t)
end

## API methods
isexact(elm::EIELM) = elm.p == elm.s

function input_to_node(elm::EIELM, x::AbstractArray, Wt::AbstractMatrix, d)
    x*Wt .+ d'
end

function hidden_out(elm::EIELM, x::AbstractArray, Wt::AbstractMatrix=elm.At, d=elm.b)
    elm.activation(input_to_node(elm, x, Wt, d))
end


function fit!(elm::EIELM, x::AbstractArray, t::AbstractArray)
    # Step 1: Initialization
    L = 0
    E = copy(t)
    min_Ei = copy(E)

    # Step 2: learning
    while L < elm.Lmax && norm(E) > elm.ϵ
        L += 1
        min_Ei_contrib = Inf

        for i in 1:elm.k
            Ai = 2*rand(elm.q, 1) - 1  # uniform [-1, 1]
            bi = 2*rand() - 1          # uniform [-1, 1]
            Hi = hidden_out(elm, x, Ai, bi)
            βi = dot(E, Hi) / dot(Hi, Hi)
            Ei = E - βi*Hi
            Ei_contrib = norm(Ei)

            if Ei_contrib < min_Ei_contrib
                elm.At[:, L] = Ai
                elm.b[L] = bi
                elm.β[L] = βi
                min_Ei_contrib = Ei_contrib
                copy!(min_Ei, Ei)
            end

        end

        copy!(E, min_Ei)
    end

    # chop before returning
    elm.At = elm.At[:, 1:L]
    elm.b = elm.b[1:L]
    elm.β = elm.β[1:L]
    elm
end

function (elm::EIELM)(x′::AbstractArray)
    @assert size(x′, 2) == elm.q "wrong input dimension"
    return hidden_out(elm, x′) * elm.β
end

function Base.show{TA}(io::IO, elm::EIELM{TA})
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
