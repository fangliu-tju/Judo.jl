# step08

mutable struct Variable
    value
    grad
    creator
end
Variable(data) = Variable(data, nothing, nothing)
Variable() = Variable(nothing, nothing, nothing)

abstract type Func end  
function (f::Func)(fun::Function, input::Variable)
    x = input.value       
    y = fun(x) 
    output = Variable(y)           
    f.input = input       
    f.output = output          
    setcreator!(output, f)
    #output.creator = f   
    return output      
end


setcreator!(v::Variable, func::Func) = v.creator = func

# Square
mutable struct Square <: Func 
    input  
    output 
end
Square() = Square(nothing, nothing) 
(f::Square)(x::Variable) = f(x) do x
    x.^2
end
function ∇(f::Square, gy)  
    x_value = f.input.value 
    2 .* x_value .* gy      
end

# Exp
mutable struct Exp <: Func
    input
    output
end
Exp() = Exp(nothing, nothing)
(f::Exp)(x::Variable) = f(x) do x 
    exp.(x)
end
function ∇(f::Exp, gy)          
    x_value = f.input.value
    exp.(x_value) .* gy
end

# 以循环方式实现了函数的求导
function gradient!(v::Variable)
    funcs = Func[] # 需要预置一个抽象函数类型的数组， 各函数本身类型不同， 但有相同的父类型
    f = v.creator     
    isnothing(f) && return # 如果变量不是通过函数计算得到的， 直接返回 `nothing`
    push!(funcs, f)        # 将创造者压入数组
    while !isempty(funcs)
        f = pop!(funcs)    # 获取函数（创造者）
        x, y = f.input, f.output  # 获取函数的输入输出
        x.grad = ∇(f, y.grad)     # 求传播到 `x` 的梯度
        !isnothing(x.creator) && push!(funcs, x.creator) # 如果 `x` 是中间变量， 继续向下传播
    end
end

# main
A, B, C = Square(), Exp(), Square()
x = Variable([0.5])
a = A(x)
b = B(a)
y = C(b)

y.grad = [1.0]
gradient!(y)
println(x.grad)