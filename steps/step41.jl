# step41

using Judo

function main()
    W = Variable(randn(2, 3))
    x = Variable(randn(3, 4))
    y = W ⋅ x
    @inference gradient!(y)
    @show size(W.grad)
    @show size(x.grad)
end

@time main();