# step04

mutable struct Variable
    value
end

abstract type Func end  
function (f::Func)(fun::Function, input::Variable) 
    x = input.value  # 取出变量中的值， 共性操作
    y = fun(x)       # 对变量值进行函数运算， 特性操作
    Variable(y)      # 将函数结果转换成 `Variable` 类型， 共性操作
end

# Square
mutable struct Square <: Func end
(f::Square)(x::Variable) = f(x) do x
    x.^2
end

# Exp
mutable struct Exp <: Func end
(f::Exp)(x::Variable) = f(x) do x
    exp.(x)
end

# 新增代码, 实现数值微分
function numerical_diff(f, x::Variable, eps=1e-4)
    x0 = Variable(x.value .- eps)
    x1 = Variable(x.value .+ eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.value .- y0.value) ./ 2eps 
end

# main
f = Square()
x = Variable([2.0])
dy = numerical_diff(f, x)
println(dy)


function g(x)
    A, B, C = Square(), Exp(), Square()
    return C(B(A(x)))
end

x = Variable([0.5])
dy = numerical_diff(g, x)
println(dy)