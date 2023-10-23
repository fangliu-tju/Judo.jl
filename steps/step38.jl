# step38

using Judo

function main()
    x = Variable([1 3 5; 2 4 6])
    y = reshape(x,(6,))
    @show y
    @inference gradient!(y,retain_grad=true)
    @show y.grad
    @show x.grad
    x = Variable(randn(1,2,3))
    @show y = reshape(x,[2,3])
    @show y = reshape(x,2,3)
end

main();