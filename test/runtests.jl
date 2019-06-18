using SLFN
using Test
using Random

const TS = (AlgebraicNetwork, ELM, OSELM, ROSELM, TSELM)
const verbose = true

Random.seed!(42)
x = range(0, stop=1, length=50) + 0.1*randn(50); y = sin.(6x)
for T in TS
    n = T(x, y; s=30)
    verbose && println(typeof(n), " 1")
    @test maximum(abs.(n(x) - y)) < 1e-2
end

# multi-dim version
nx, ny, nz = 10, 8, 9
x = range(0, stop=1, length=nx)
y = range(0, stop=1, length=ny)
z = range(0.1, stop=1, length=nz)

f1(x) = @. sin(2pi/nx * (x-1))
f2(x, y) = @. f1(x) .* cos(4pi/ny * (y-1))
f3(x, y, z) = @. f2(x, y) .* log(z)

# 2d case
X2 =[kron(ones(ny), x) kron(y, ones(nx))]
Y2 = f2(X2[: ,1], X2[:, 2])

for T in TS
    n = T(X2, Y2, s=40)
    verbose && println(typeof(n), " 2")
    @test maximum(abs.(n(X2) - Y2)) < 1e-1
end

X3 =[kron(ones(ny*nz), x) kron(ones(nz), y, ones(nx)) kron(z, ones(nx*ny))]
Y3 = f3(X3[:, 1], X3[:, 2], X3[:, 3])

for T in TS
    n = T(X3, Y3, s=200)
    verbose && println(typeof(n), " 3")
    @test maximum(abs.(n(X3) - Y3)) < 1e-1
end

# test non [0, 1] domain works (i.e. standardizing is working)
x = 150 * rand(5, 5)
y = sin.(range(0, stop=6, length=size(x, 1)))

for T in TS
    n = T(x, y, s=size(x, 1))
    verbose && println(typeof(n), " standardizing")
    @test maximum(abs.(n(x) - y)) < 1e-10
end
