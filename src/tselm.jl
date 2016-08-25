"""

#### References

Two-stage extreme learning machine for regression.

Yuan Lan, Yeng Chai Soh, and Guang-Bin Huang.

Neurocomputing, 2010 vol. 73 (16-18) pp. 3028-3038.

http://linkinghub.elsevier.com/retrieve/pii/S0925231210003401
"""
type TSELM{TA<:AbstractActivation,TN<:AbstractNodeInput,TV<:AbstractArray{Float64}} <: AbstractSLFN
    p::Int  # Number of training points
    q::Int  # Dimensionality of function domai
    s::Int  # maximum number of neurons
    ngroup::Int  # number of groups
    npg::Int  # nodes per group
    activation::TA
    neuron_type::TN
    Wt::Matrix{Float64}
    d::Vector{Float64}
    v::TV

    function TSELM(p::Int, q::Int, s::Int, ngroup::Int, npg::Int, activation::TA,
                   neuron_type::TN)
        new(p, q, s, ngroup, npg, activation, neuron_type)
    end
end


function TSELM{TA<:AbstractActivation,
               TN<:AbstractNodeInput,
               TV<:AbstractArray}(x::AbstractArray, y::TV; activation::TA=Sigmoid(),
                                  neuron_type::TN=Linear(), s::Int=size(x, 1),
                                  ngroup::Int=5,
                                  npg::Int=ceil(Int, size(x, 1)/10),
                                  reg::AbstractLinReg=LSSVD())
    q = size(x, 2)  # dimensionality of function domain
    p = size(x, 1)  # number of training points
    s = min(p, s)  # can't have more neurons than obs
    npg = min(ceil(Int, s/4), npg)  # ensure at least 4 groups
    out = TSELM{TA,TN,TV}(p, q, s, ngroup, npg, activation, neuron_type)
    fit!(out, x, y, reg)
end

## helper methods
# take every other data point so that when a is something like
# a linspace we still get decent coverage of the whole domain
_split_data(a::AbstractVector) = (a[1:2:end], a[2:2:end])
_split_data(a::AbstractMatrix) = (a[1:2:end, :], a[2:2:end, :])

function forward_selection!(elm::TSELM, x::AbstractArray, y::AbstractVector,
                            reg::AbstractLinReg)
    # split data sets
    train_x, validate_x = _split_data(x)
    train_y, validate_y = _split_data(y)
    N = size(train_x, 1)
    N_validate = size(validate_x, 1)

    # initialize
    L = 0
    R = eye(N, N)
    W = zeros(elm.q, elm.s + elm.npg)
    d = zeros(elm.s + elm.npg)

    while L < elm.s
        L += elm.npg
        inds = (L-elm.npg)+1:L

        # variables to hold max ΔJ and corresponding δH for this
        # batch of groups. Define empty matrices to get type stability
        ΔJ = -Inf
        ΔR = zeros(0, 0)
        Ak = zeros(0, 0)
        dk = zeros(0)

        for i in 1:elm.ngroup
            # Step 1: randomly generate hidden parameters for this group
            wiT = 2*rand(elm.q, elm.npg) - 1  # uniform [-1, 1]
            di = rand(elm.npg)                # uniform [0, 1]

            # Step 2: generate the hidden output for this group
            δHi = hidden_out(elm, train_x, wiT, di)

            # Step 3: compute the contribution of this group to cost function
            # TODO: sometimes (δHi'R*δHi) is singular. I can loop over above until
            #       it works
            ΔRi = (R*δHi)*(pinv(δHi'R*δHi) * (δHi'R'))
            ΔJi = dot(train_y, ΔRi*train_y)

            # Step 4: keep this group if it is the best we've seen so far
            if ΔJi > ΔJ
                ΔR = ΔRi
                Ak = wiT
                dk = di
            end
        end

        # Step 5: Update hidden node parameters (A, b), R matrix, H
        W[:, inds] = Ak
        d[inds] = dk
        R -= ΔR
    end

    # Step 6: Find the optimal number of neurons pstar
    pstar = 0
    fpe_min = Inf
    for p in elm.npg:elm.npg:elm.s
        Hp = hidden_out(elm, validate_x, W[:, 1:p], d[1:p])
        βp = regress(reg, Hp, validate_y)
        SSEp = StableReg.should_add_intercept(reg) ?
            norm(validate_y - (Hp*βp[2:end] + βp[1]), 2) :
            norm(validate_y - (Hp*βp), 2)
        fpe = SSEp/N_validate * (N_validate+p)/(N_validate-p)

        if fpe < fpe_min
            fpe_min = fpe
            pstar = p
        end
    end

    # Step 7: Rebuild the net with the selected neurons and fit
    #         with entire training set
    elm.Wt = W[:, 1:pstar]
    elm.d = d[1:pstar]
    H = hidden_out(elm, x, elm.Wt, elm.d)
    elm.v = regress(reg, H, y)

    elm, H
end

function backward_elimination!(elm::TSELM, x::AbstractArray, y::AbstractVector,
                               H::AbstractMatrix)
    # TODO: come back to this
    return
    pstar = size(elm.Wt, 2)
    Hr = copy(H)
    Hpstar = H[:, pstar]
    N = size(H, 1)

    y = H*elm.v

    # Step 1: Compute press1 for all nodes
    press1 = zeros(Float64, pstar -1)
    for k in pstar-1:-1:1
        copy!(Hr, H)
        Hk = view(H, :, k)
        Hr[:, pstar] = Hk
        Hr[:, k] = Hpstar

        ϵ = y - Hr*elm.v
        M = inv(Hr'Hr)

        for i in 1:N
            hri = Hr[i, :]
            ϵi = ϵ[i] / (1 - dot(hri, M*hri))
            press1[k] += ϵi^2
        end
        press1[k] /= N

    end

end

function fit!(elm::TSELM, x::AbstractArray, y::AbstractVector,
              reg::AbstractLinReg)
    elm, H = forward_selection!(elm, x, y, reg)
    backward_elimination!(elm, x, y, H)
    elm
end

function Base.show{TA,TN}(io::IO, elm::TSELM{TA,TN})
    s =
    """
    TSELM with
      - $(TA) Activation function
      - $(TN) Neuron type
      - $(elm.q) input dimension(s)
      - $(size(elm.Wt, 2)) neuron(s)
      - $(elm.p) training point(s)
      - Algorithm parameters:
          - $(elm.ngroup) trials per group
          - $(elm.npg) neurons per group
          - $(elm.s) max neurons
    """
    print(io, s)
end
