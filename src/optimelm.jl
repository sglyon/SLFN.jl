"""
This method is inspired by (but not an implementation of) the method found
in

Nelder-mead enhanced extreme learning machine.

Philip Reiner and Bogdan M Wilamowski.

2013 IEEE 17th International Conference on Intelligent Engineering Systems
(INES), 2013 pp. 225-230.

http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=6632816

"""
type OptimELM{TV<:AbstractArray{Float64},TN<:RBF} <: AbstractSLFN
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domain
    s::Int  # number of neurons
    ϵ::Float64  # expected learning accuracy

    # internal parameters
    Wt::Matrix{Float64}  # transpose of W matrix
    d::Vector{Float64}
    neuron_type::TN
    v::TV
    E::TV  # running error


    function OptimELM(p::Int, q::Int, s::Int, ϵ::Float64, neuron_type::TN)
        Wt = 2*rand(q, s) - 1
        d = rand(s)
        v = TV(s)
        new(p, q, s, ϵ, Wt, d, neuron_type, v)
    end
end

function OptimELM{TV<:AbstractArray,TN<:RBF}(x::AbstractArray, y::TV;
                                           s::Int=size(y, 1), ϵ::Float64=1e-6,
                                           neuron_type::TN=RBF(Gaussian()))
    q = size(x, 2)  # dimensionality of function domain
    p = size(y, 1)  # number of training points
    s = min(p, s)   # Can't have more neurons than training points
    out = OptimELM{TV,TN}(p, q, s, ϵ, neuron_type)
    out.E = copy(y)
    fit!(out, x, y)
end

function fit!(elm::OptimELM, x::AbstractArray, y::AbstractArray)
    # ------- Step 1: initialize
    n = 0
    err = Inf
    xt = x'
    E = elm.E

    # ------- Step 2: Learning
    while n < elm.s && err > elm.ϵ
        # --- Step 2.a: Increase n
        n += 1

        # --- Step 2.b: find index of max error
        j = indmax(abs(E))

        # --- Step 2.c: Assign new center to be input x
        cn = xt[:, j]
        elm.Wt[:, n] = cn

        # --- Step 2.d: Initialze β = Eⱼ
        βn = E[j]

        # --- Step 2.e: define objective function
        function obj(σ)
            sse = 0.0
            for i in 1:elm.p
                gi = hidden_out(elm.neuron_type, view(xt, :, i), cn, σ)
                sse += (E[i] - βn * gi)^2
            end
            sse
        end

        # Step 2.f: do some iterations of an optimization algorithm
        # call to elm.k nm simplex iterations (or call Optim??)
        res = Optim.optimize(obj, 1e-5, 1e2)
        elm.d[n] = Optim.minimizer(res)

        # Step 2.g: re-calculate vn
        num = 0.0
        denom = 0.0
        for i in 1:elm.p
            gp = hidden_out(elm.neuron_type, view(xt, :, i), cn, elm.d[n])
            num += E[i] * gp
            denom += gp*gp
        end
        elm.v[n] = num/denom

        # Step 2.h: update error trace
        for i in 1:elm.p
            E[i] -= elm.v[n] * hidden_out(elm.neuron_type, view(xt, :, i), cn, elm.d[n])
        end

        # Step 2.i: update err
        err = maxabs(E)

    end

    # trim in case we achieved desired accuracy before adding elm.s neurons
    if n < elm.s
        elm.v = elm.v[1:n]
        elm.d = elm.d[1:n]
        elm.Wt = elm.Wt[:, 1:n]
    end

    return elm

end

function Base.show{TA,TN}(io::IO, elm::OptimELM{TA,TN})
    m = match(r"SLFN.RBF\{SLFN\.(.+),Distances\.(.+)\}", string(TN))
    ta, td = m[1], m[2]
    n = size(elm.v, 1)
    s = """
    OptimELM with
      - $(elm.q) input dimension(s)
      - $n RBF{$ta,$td} neuron$(n>1 ? "s": "")
      - $(elm.p) training point(s)
    """
    print(io, s)
end
