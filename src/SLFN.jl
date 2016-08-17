__precompile__()

module SLFN

abstract AbstractSLFN

export AbstractActivation, Sigmoid, SoftPlus, Tanh, Relu, AbstractSLFN,
    fit!, isexact, input_to_node, hidden_out, ELM, TSELM, EIELM,
    AlgebraicNetwork

const WARNINGS = [false]
dont_warn_me() = (WARNINGS[1] = false; return)
warn_me() = (WARNINGS[1] = true; return)

include("activations.jl")
include("elm.jl")
include("tselm.jl")
include("eielm.jl")
include("algebraic.jl")


end # module
