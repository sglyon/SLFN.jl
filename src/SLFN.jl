__precompile__()

module SLFN

using Distances, Compat
using Compat: view

include("stable_regression.jl")

using .StableReg
export StableReg, AbstractLinReg, AbstractLSMethod, AbstractLADMethod,
    OLS, LSSVD, LSLdiv, RLSTikhonov, RLST, LADPP, LADDP,
    RLADPP, RALDDP, RLSSVD, regress

abstract AbstractSLFN

export AbstractActivation, Sigmoid, SoftPlus, Tanh, Relu, AbstractSLFN,
    fit!, isexact, input_to_node, hidden_out,
    AlgebraicNetwork, ELM, TSELM, EIELM, OSELM, ROSELM,
    Linear, RBF, Gaussian

const WARNINGS = [false]
dont_warn_me() = (WARNINGS[1] = false; return)
warn_me() = (WARNINGS[1] = true; return)

isexact(elm::AbstractSLFN) = false

function input_to_node(elm::AbstractSLFN, x, Wt=elm.Wt, d=elm.d)
    input_to_node(elm.neuron_type, x, Wt, d)
end

function hidden_out(elm::AbstractSLFN, x, Wt=elm.Wt, d=elm.d)
    elm.activation(input_to_node(elm, x, Wt, d))
end

function hidden_out(elm::AbstractSLFN, x::Number, Wt=elm.Wt, d=elm.d)
    elm.activation(input_to_node(elm, x, Wt, d))
end

include("node_inputs.jl")
include("activations.jl")
include("elm.jl")
include("oselm.jl")
include("tselm.jl")
include("eielm.jl")
include("roselm.jl")
include("algebraic.jl")

for T in subtypes(AbstractSLFN)
    @eval @compat function (elm::$(T))(x′)
        @assert size(x′, 2) == elm.q "wrong input dimension"
        if length(elm.v) == size(elm.Wt, 2) + 1 # has intercept
            hidden_out(elm, x′) * elm.v[2:end] + elm.v[1]
        else
            hidden_out(elm, x′) * elm.v
        end
    end

    @eval @compat function (elm::$(T))(x′::Number)
        if length(elm.v) == size(elm.Wt, 2) + 1 # has intercept
            out_vec = hidden_out(elm, x′) * elm.v[2:end] + elm.v[1]
        else
            out_vec = hidden_out(elm, x′) * elm.v
        end

        @assert length(out_vec) == 1 "There's a bug -- file an issue"
        return out_vec[1]
    end
end

end # module
