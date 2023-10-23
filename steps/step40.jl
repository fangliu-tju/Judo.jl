# step 40

using Judo

function main()
    x = Variable([1, 2, 3])
    y = broadcastto(x, (3, 2))
    @show y.value
    x = Variable([1 2 3; 4 5 6])
    y = sumto(x, (1,3))
    @show y.value
    @show size(y)

    x1 = Variable([1,2,3])
    x2 = Variable(10)
    y = x1 + x2
    @show y
    @inference gradient!(y)
    @show x2.grad;
end

@time main()