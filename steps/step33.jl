# step33

using Judo 

f(x) = x^4 - 2 * x^2
#=
x = Variable([2.0])
y = f(x)
gradient!(y)
println(x.grad)

gx = x.grad
cleargrad!(x)
@inference gradient!(gx)
println(x.grad)
=#

function main(f,init=Variable([0.0]))
    x = init
    iters= 9
    for i in 0:iters
        println(i, " ", x)
        y = f(x)      # 每次循环， `y` 都是一个全新的值
        cleargrad!(x) # `x` 是老变量， 里面保存了前面计算步骤的信息
        gradient!(y)

        gx = x.grad   # `x` 的梯度也是 `x` 的函数， 本质上与 `y` 没有区别
        cleargrad!(x) # 反向传播后， 产生了梯度, 下次计算前需要清理
        @inference gradient!(gx) # 不需要计算高阶梯度了
        gx2 = x.grad
        x.value -= gx.value ./ gx2.value # 只更新 `x` 的值， 提高计算效率
    end
end

@time main(f, Variable(2.0))

