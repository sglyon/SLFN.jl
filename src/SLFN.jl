__precompile__()

module SLFN

using Distances

abstract AbstractSLFN

export AbstractActivation, Sigmoid, SoftPlus, Tanh, Relu, AbstractSLFN,
    fit!, isexact, input_to_node, hidden_out, ELM, TSELM, EIELM,
    Linear, RBF, Gaussian,
    AlgebraicNetwork

const WARNINGS = [false]
dont_warn_me() = (WARNINGS[1] = false; return)
warn_me() = (WARNINGS[1] = true; return)

include("node_inputs.jl")
include("activations.jl")
include("elm.jl")
include("tselm.jl")
include("eielm.jl")
include("algebraic.jl")


end # module
