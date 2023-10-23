# step37

using Judo

function main()
    x = Variable([1 2 3; 4 5 6])
    c = Variable([10 20 30; 40 50 60])
    t = x + c
    y = sum(t)
    @inference gradient!(y,retain_grad=true)
    println(y.grad)
    println(t.grad)
    println(x.grad)
    println(c.grad)
end

@time main()