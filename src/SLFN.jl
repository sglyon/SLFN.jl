__precompile__()

module SLFN

using Distances, Compat, Optim
using Compat: view

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

function hidden_out(elm::AbstractSLFN, x::AbstractArray, Wt=elm.Wt, d=elm.d)
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
include("nmeelm.jl")


end # module
