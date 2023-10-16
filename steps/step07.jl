# step07

# 对 `Variable` 类型进行改造， 增加创造者（creator）域
mutable struct Variable
    value
    grad
    creator
end
Variable(data) = Variable(data, nothing, nothing)
Variable() = Variable(nothing, nothing, nothing)

abstract type Func end  
function (f::Func)(fun::Function, input::Variable)
    x = input.value       # 取出变量中保存的值
    y = fun(x)            # 函数求值运算， `fun` 是普通函数
    f.input = input       # 保存函数的输入， `Variable` 类型
    output = Variable(y)  # 将输出转换成 `Variable` 类型
    f.output = output     # 保存函数的输出， `Variable` 类型
    
    setcreator!(output, f)# 输出的创造者是函数 `f`， 实际是可调用数据类型
    #output.creator = f   # 直接赋值也是可以的， 与上一行的语句二选一
    return output      
end

# 定义一个函数， 用于设置变量的创造函数， 不定义这个函数也是可以的， 后面看看是不是要取消这个函数
setcreator!(v::Variable, func::Func) = v.creator = func

# Square
mutable struct Square <: Func 
    input  
    output # `Square` 类型中增加 `output` 域， 用于保存输出变量
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

# 以递归方式实现了函数的求导， 可以由指定的输出变量自动计算得到输入变量的导数
function gradient!(v::Variable)
    f = v.creator             # 取出变量的创造者（函数）
    if !isnothing(f)          # 当前变量有创造者时， 进行下面的计算
        x = f.input           # 取出函数的输入变量
        x.grad = ∇(f, v.grad) # 对输入变量求局部导数与上游传来的导数的乘积
        gradient!(x)          # 进入下一个传播链， 再次求导
    end
end

# main
A, B, C = Square(), Exp(), Square()
x = Variable([0.5])
a = A(x)
b = B(a)
y = C(b)

@assert y.creator == C
@assert y.creator.input == b
@assert y.creator.input.creator == B
@assert y.creator.input.creator.input == a
@assert y.creator.input.creator.input.creator == A
@assert y.creator.input.creator.input.creator.input == x

y.grad = [1.0]
gradient!(y)
println(x.grad)