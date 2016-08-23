using SLFN
using Base.Test

const TS = (AlgebraicNetwork, ELM, OSELM, ROSELM, TSELM)
const verbose = true

let
    srand(42)
    x = linspace(0, 1, 50) + 0.1*randn(50)
    y = sin(6x)
    for T in TS
        verbose && println(T, 1)
        n = T(x, y; s=30)
        @test maxabs(n(x) - y) < 1e-2
    end
end

# multi-dim version
let
    nx, ny, nz = 10, 8, 9
    x = linspace(0, 1, nx)
    y = linspace(0, 1, ny)
    z = linspace(0.1, 1, nz)

    f1(x) = sin(2pi/nx * (x-1))
    f2(x, y) = f1(x) .* cos(4pi/ny * (y-1))
    f3(x, y, z) = f2(x, y) .* log(z)

    # 2d case
    X2 =[kron(ones(ny), x) kron(y, ones(nx))]
    Y2 = f2(X2[: ,1], X2[:, 2])

    for T in TS
        verbose && println(T, 2)
        n = T(X2, Y2, s=40)
        @test maxabs(n(X2) - Y2) < 1e-2
    end

    X3 =[kron(ones(ny*nz), x) kron(ones(nz), y, ones(nx)) kron(z, ones(nx*ny))]
    Y3 = f3(X3[:, 1], X3[:, 2], X3[:, 3])

    for T in TS
        verbose && println(T, 3)
        n = T(X3, Y3, s=200)
        @test maxabs(n(X3) - Y3) < 1e-2
    end
end
