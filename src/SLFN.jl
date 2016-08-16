__precompile__()

module SLFN

abstract AbstractSLFN

export AbstractActivation, Sigmoid, SoftPlus, Tanh, Relu, AbstractSLFN,
       fit!, isexact, input_to_node, hidden_out, ELM, AlgebraicNetwork

include("activations.jl")
include("elm.jl")
include("algebraic.jl")

end # module
