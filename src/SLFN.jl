__precompile__()

module SLFN

using Distances, Compat
using Compat: view
import Optim

include("stable_regression.jl")

using .StableReg
export StableReg, AbstractLinReg, AbstractLSMethod, AbstractLADMethod,
    OLS, LSSVD, LSLdiv, RLSTikhonov, RLST, LADPP, LADDP,
    RLADPP, RALDDP, RLSSVD, regress

abstract AbstractSLFN

export AbstractActivation, Sigmoid, SoftPlus, Tanh, Relu, AbstractSLFN,
    fit!, isexact, input_to_node, hidden_out, n_coefs, activate, activate!,
    AlgebraicNetwork, ELM, TSELM, EIELM, OSELM, ROSELM, LSIELM,
    Linear, RBF, Gaussian

const WARNINGS = [false]
dont_warn_me() = (WARNINGS[1] = false; return)
warn_me() = (WARNINGS[1] = true; return)

isexact(elm::AbstractSLFN) = false

function input_to_node(elm::AbstractSLFN, x, Wt=elm.Wt, d=elm.d)
    input_to_node(elm.neuron_type, x, Wt, d)
end

function hidden_out(elm::AbstractSLFN, x, Wt=elm.Wt, d=elm.d)
    hidden_out(elm.neuron_type, x, Wt, d)
end

n_coefs(elm::AbstractSLFN) = length(elm.Wt) + length(elm.d) + length(elm.v)

include("activations.jl")
include("node_inputs.jl")
include("elm.jl")
include("oselm.jl")
include("tselm.jl")
include("eielm.jl")
include("roselm.jl")
include("algebraic.jl")
include("lsielm.jl")
include("optimelm.jl")

StableReg.standardize(x::AbstractMatrix, μ::AbstractVector, σ::AbstractVector) =
    (x .- μ') ./ σ'

StableReg.standardize(x::AbstractVector, μ::AbstractVector, σ::AbstractVector) =
    length(μ) == 1 ? (x .- μ') ./ σ' : (x .- μ) ./ σ

function StableReg.standardize(x::Number, μ, σ)
    length(μ) == length(σ) == 1 || error("x shouldn't be scalar")
    (x - μ[1]) / σ[1]
end

for T in subtypes(AbstractSLFN)
    @eval @compat function (elm::$(T))(_x′)
        @assert size(_x′, 2) == elm.q "wrong input dimension"
        x′ = isdefined(elm, :μx) ? standardize(_x′, elm.μx, elm.σx) : _x′
        if length(elm.v) == size(elm.Wt, 2) + 1 # has intercept
            hidden_out(elm, x′) * elm.v[2:end] + elm.v[1]
        else
            hidden_out(elm, x′) * elm.v
        end
    end

    @eval @compat function (elm::$(T))(_x′::Number)
        x′ = isdefined(elm, :μx) ? standardize(_x′, elm.μx, elm.σx) : _x′
        if length(elm.v) == size(elm.Wt, 2) + 1 # has intercept
            out_vec = hidden_out(elm, x′) * elm.v[2:end] + elm.v[1]
        else
            out_vec = hidden_out(elm, x′) * elm.v
        end

        @assert length(out_vec) == 1 "There's a bug -- file an issue"
        out_vec[1]
    end
end

Base.getindex(elm::AbstractSLFN, x) = elm(x)

end # module
