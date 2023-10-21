# step26

include("../src/core_simple.jl")
include("../src/utils.jl") 

goldstein(x, y) = (1 + (x + y + 1)^2 * (19 - 14 * x + 3 * x^2 - 14 * y + 6 * x * y + 3 * y^2)) * (30 + (2 * x - 3 * y)^2 * (18 - 32 * x + 12 * x^2 + 48 * y - 36 * x * y + 27 * y^2))

function main()
    x = Variable([1.0], name="x")
    y = Variable([1.0], name="y")
    z = goldstein(x, y)
    z.name = "z"

    #gradient!(z) # 计算图是在求值时建立的， 与求导与否无关
    plot_dot_graph(z, verbose = false, file="images/goldstein.png")
end

@time main()