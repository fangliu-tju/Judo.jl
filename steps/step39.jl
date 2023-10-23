# step 39

using Judo

function main()
    x = Variable([1, 2, 3, 4, 5, 6])
    y = sum(x)
    @inference gradient!(y)
    println(y)
    println(x.grad)

    x = Variable([1 2 3;4 5 6])
    y = sum(x,dims=1)
    println(y)
    println(size(x), " -> ", size(y))
end

@time main()