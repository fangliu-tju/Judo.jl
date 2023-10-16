# step06

# 对 `Variable` 类型进行改造， 增加导数（grad）域
mutable struct Variable
    value
    grad
end
# 两个外部构造函数，分别对应一个输入参数和零个输入参数
Variable(data) = Variable(data, nothing)
Variable() = Variable(nothing, nothing)

abstract type Func end  
function (f::Func)(fun::Function, input::Variable)
    f.input = input  # 将函数的输入变量保存在函数中， 共性操作
    x = input.value  
    y = fun(x)       
    Variable(y)      
end

# Square
mutable struct Square <: Func 
    input  # `Square` 类型中增加 `input` 域， 用于保存输入变量
end
Square() = Square(nothing) # 定义零个参数的外部构造函数
(f::Square)(x::Variable) = f(x) do x
    x.^2
end
# 定义 `Square` 函数的导数计算公式, 在这步上没有定义新的数据结构是因为不需要， 后面会更加清晰
function ∇(f::Square, gy)   # `∇` 的输入方法是 `\nabla tab`， 表示对函数（第一个参数）求导 
    x_value = f.input.value # 计算导数的位置， 即 `x` 值， 保存在函数的输入域中
    2 .* x_value .* gy      # `gy` 是上游传来的导数， 数组类型， 后面会变成 `Variable` 类型
end

# Exp
mutable struct Exp <: Func
    input
end
Exp() = Exp(nothing)
(f::Exp)(x::Variable) = f(x) do x # 定义 `Exp` 所对应的求值计算（正向传播）
    exp.(x)
end
# 定义 `Epx` 函数的导数计算公式
function ∇(f::Exp, gy)            # 定义 `Exp` 所对应的求导计算（反向传播）
    x_value = f.input.value
    exp.(x_value) .* gy
end

# main
A, B, C = Square(), Exp(), Square()
x = Variable([0.5])

# 求值
a = A(x)
b = B(a)
y = C(b)

# 求导
y.grad = [1.0]
b.grad = ∇(C, y.grad)
a.grad = ∇(B, b.grad)
x.grad = ∇(A, a.grad)

println(x.grad)