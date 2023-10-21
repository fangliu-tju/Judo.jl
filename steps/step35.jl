# step35

using Judo 

function main()
    x = Variable([1.0], name="x")
    y = tanh(x)
    y.name = "y"
    gradient!(y)

    iters = 1
    for i in 1:(iters - 1)
        println(i)
        gx = x.grad
        cleargrad!(x)
        gradient!(gx)
    end
    gx = x.grad
    gx.name = "gx$(iters)"

    cd("images")
    println("plotting graph, please waiting...")
    plot_dot_graph(gx, file="tanh$(iters).png")
    cd("..")
    println("done!")
end

@time main()