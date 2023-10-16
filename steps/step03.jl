# step03

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

# 新增代码, 采用匿名函数的方式
# Exp
mutable struct Exp <: Func end
(f::Exp)(x::Variable) = f(x) do x
    exp.(x)
end

# main
A, B, C = Square(), Exp(), Square()
x = Variable([0.5])
a = A(x)
b = B(a)
y = C(b)

println(typeof(y))
println(y.value)