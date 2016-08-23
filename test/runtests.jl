using SLFN
using Base.Test

let
    srand(42)
    x = linspace(0, 1, 50) + 0.1*randn(50)
    y = sin(6x)
    for T in (AlgebraicNetwork, ELM, OSELM, ROSELM, TSELM)
        n = T(x, y; s=30)
        @test maxabs(n(x) - y) < 1e-2
    end
end
