using SLFN

function gridmake(x, y, z)
    nx, ny, nz = length(x), length(y), length(z)
    hcat(repeat(x, outer=[ny*nz]),
         repeat(y, outer=[nz], inner=[nx]),
         repeat(z, inner=[nx*ny]))
end

function main(new_data=:same)
    println("Using $(new_data) data on each online training epoch")

    # create some data
    nx, ny, nz = 10, 8, 9
    x = linspace(0, 1, nx)
    y = linspace(0, 1, ny)
    z = linspace(0.1, 1, nz)

    # create some arbitrary functions
    f1(x) = sin(2pi/nx * (x-1))
    f2(x, y) = f1(x) .* cos(4pi/ny * (y-1))
    f3(x, y, z) = f2(x, y) .* log(z)

    # construct grids on the data
    X = gridmake(x, y, z)
    Y = f3(X[:, 1], X[:, 2], X[:, 3])

    # in the next line we train a single hidden layer neural network with tanh
    # neurons and random hidden parameters. The hidden neurons are assigned
    # random parameters such that on the initial training data
    elm = ROSELM(X, Y, s=60, neuron_type=Linear(Tanh()))

    # see how well it generalizes
    x2 = linspace(0, 1, nx+5)
    y2 = linspace(0, 1, ny+5)
    z2 = linspace(0.1, 1, nz+5)

    X2 = gridmake(x2, y2, z2)
    Y2 = f3(X2[:, 1], X2[:, 2], X2[:, 3])

    function myprint(elm, X, Y, it)
        @printf("%7i | %8.3f | %8.3f\n",  it, maxabs(elm(X2) - Y2), norm(elm.v))
    end

    @printf("%7s | %8s | %8s\n", "init", "max err", "norm(v)")

    myprint(elm, X2, Y2, 0)

    for i in 1:40
	      for _ in 1:5
            if new_data == :same
	              SLFN.fit!(elm, X, Y)
            elseif new_data == :rand
                # generate random data on the same grid
                _X = gridmake(rand(nx), rand(ny), 0.9*rand(nz)+0.1)
                _Y = f3(_X[:, 1], _X[:, 2], _X[:, 3])
                SLFN.fit!(elm, _X, _Y)
            end
	      end
        myprint(elm, X2, Y2, i)
    end

end

main(:same)

println("\n\n")
main(:rand)
