# step02

mutable struct Variable
    value
end

# 深度学习中的函数不是普通的函数， 需要为它设计一种数据类型
# 为了实现代码复用， 需要定义抽象数据类型来完成共性（每个函数类型中都要重复）的操作
abstract type Func end  

# 定义了可调用数据类型的一般调用方法， 把函数类型的参数作为第一个参数， 目的是后面方便使用匿名函数
function (f::Func)(fun::Function, input::Variable) 
    x = input.value  # 取出变量中的值， 共性操作
    y = fun(x)       # 对变量值进行函数运算， 特性操作
    Variable(y)      # 将函数结果转换成 `Variable` 类型， 共性操作
end

# 定义一个求平方的函数， 实际上是一种数据类型， 现在来看不是必要的， 但后面会有扩展
mutable struct Square <: Func end

# 为 `Square` 实现调用方法， 有两种方法
# 第一种方法， 先定义一个普通函数， 再在可调用数据类型中使用这个函数
square(x) = x.^2  # 定义一个普通函数， 作为 `Square` 数据类型的第一个输入参数
(f::Square)(x::Variable) = f(square, x) # 定义单参数的调用方法， 使用了普通函数 `square`
# `f(square, x)` 实际上是调用了 `(f::Func)(fun::Function,  input::Variable)`

#=
# 第二种方法， 使用匿名函数， 可以节省函数名称
(f::Square)(x::Variable) = f(x) do x
    x.^2
end
=#

# main
x = Variable([10.0, 5])
f = Square()
y = f(x)
print(typeof(y),"\n", y.value)