"""
Nelder-mead enhanced extreme learning machine.

Philip Reiner and Bogdan M Wilamowski.

2013 IEEE 17th International Conference on Intelligent Engineering Systems
(INES), 2013 pp. 225-230.

http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=6632816

"""
type NMEELM{TV<:AbstractArray{Float64}} <: AbstractSLFN
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

function NMEELM{TV<:AbstractArray}(x::AbstractArray, y::TV, s::Int=size(y, 1),
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

## Nelder-Mead -- Simplex functions
@inline centroid(x::AbstractArray) = sum(x[1:end-1]) / (length(x) - 1)
@inline reflect(elm::NMEELM, xhat::Number, xnp1::Number) = (1+elm.ρ)*xhat - elm.ρ*xnp1
@inline expand(elm::NMEELM, xhat::Number, xr::Number) = xhat + elm.χ*(xr - xhat)
@inline ocontract(elm::NMEELM, xhat::Number, xr::Number) = xhat + elm.γ*(xr - xhat)
@inline icontract(elm::NMEELM, xhat::Number, xnp1::Number) = xhat + elm.γ*(xnp1 - xhat)
@inline shrink(elm::NMEELM, x1::Number, xi::Number) = x1 + elm.α*(xi - x1)

function replace_last_sorted!(x::AbstractVector, fx::AbstractVector, xnew::Number, fxnew::Number)
    # nth position
    n = searchsortedfirst(fx, fxnew)

    # Insert x and fx into correct position
    insert!(x, n, xnew)
    insert!(fx, n, fxnew)

    # Remove last element
    pop!(x)
    pop!(fx)

    return nothing
end

function shrink_all!(x::AbstractVector, fx::AbstractVector, elm::NMEELM, f::Function)
    # Get information needed repeatedly
    x1, n = x[1], length(x)

    # Shrink all point
    for i=1:n
        x[i] = x1 + elm.α*(x[i] - x[1])
        fx[i] = f(x[i])
    end

    perm_2_sort = sortperm(fx)
    copy!(x, x[perm_2_sort])
    copy!(fx, fx[perm_2_sort])

    return nothing
end

"""
Note: Treats x and fx as if they are already sorted. This is ensured within `ksimplex`,
      but if you call this function outside of that context ensure that you are keeping
      with that restriction or weird (wrong) things could happen
"""
function singlesimplex!(x::AbstractVector, fx::AbstractVector, elm::NMEELM, f::Function)
    # Get information we will use repeatedly
    x1, xn, xnp1 = x[1], x[end-1], x[end]
    fx1, fxn, fxnp1 = fx[1], fx[end-1], fx[end]

    # First compute the centroid
    xhat = centroid(x)

    # Now compute the reflection point
    xr = reflect(elm, xhat, x[end])
    fxr = f(xr)

    # Begin obnoxious number of ifs ...
    # First check whether to just return reflected point
    if fx1 < fxr < fxn
        replace_last_sorted!(x, fx, xr, fxr)
    # If not then check the expanded point
    elseif fxr < fxn
        # Compute expansion point
        xe = expand(elm, xhat, xr)
        fxe = f(xe)

        # If fxr was smaller than second to last return smaller of reflected and
        # expanded points
        fxr < fxe ? replace_last_sorted!(x, fx, xr, fxr) : replace_last_sorted!(x, fx, xe, fxe)
    # If not reflected or expanded then check contractions
    elseif fxn < fxr < fxnp1
        # If between last 2 points then use outside contraction
        xoc = ocontract(elm, xhat, xr)
        fxoc = f(xoc)
        fxoc < fxr ? replace_last_sorted(x, fx, xoc, fxoc) : shrink_all!(x, fx)
    else
        # If larger than last point then use inside contraction
        xic = icontract(elm, xhat, xnp1)
        fxic = f(xic)

        # If better than worst point, replace it
        fxic < fnp1 ? replace_last_sorted(x, fx, xic, fxic) : shrink_all!(x, fx)
    end

    return nothing
end

function ksimplex(x::AbstractVector, elm::NMEELM, f::Function)
    # Apply function to each element of vector
    fxout = map(f, x)

    # Sort
    perm_2_sort = sortperm(fx)
    xout = x[perm_2_sort]
    copy!(fxout, fxout[perm_2_sort])

    for i=1:elm.k
        singlesimplex!(xout, fxout, elm, f)
    end

    return xout, fxout
end

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
