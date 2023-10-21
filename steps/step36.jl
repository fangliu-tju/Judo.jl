# step36

using Judo

function main()
    x = Variable([2.0])
    y = x^2
    gradient!(y)
    gx = x.grad
    cleargrad!(x)

    z = gx^3 + y
    @inference gradient!(z)
    print(x.grad)

    # 专栏中的例题
    x = Variable([1.0, 2.0])
    v = Variable([4.0, 5.0])
    y = sum(x^2)

    gradient!(y)
    gx = x.grad
    cleargrad!(x)
    z = v' ⋅ gx   # 用点乘表示矩阵乘法 
    @inference gradient!(z)
    print(x.grad)
end

@time main()