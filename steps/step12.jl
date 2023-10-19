# step12

mutable struct Variable
    value
    grad
    creator
end
Variable(data) = Variable(data, nothing, nothing)
Variable() = Variable(nothing, nothing, nothing)

abstract type Func end  
function (f::Func)(fun, inputs...) # 函数更具一般性
    xs = [input.value for input in inputs] # 遍历数组      
    ys = fun(xs...)   # 在调用普通函数时， 将数组展开成单个参数
    !isa(ys, Tuple) && (ys = tuple(ys)) # 返回值如果不是元组， 则转换成元组
    outputs = Variable.(ys) # `Variable` 是针对单个参数的， 对于数组要用到 `.` 运算符
    for output in outputs          
        setcreator!(output, f) # 每个输出都使用了相同的运算
    end         
    f.inputs = inputs       
    f.outputs = outputs
    return length(outputs) > 1 ? outputs : outputs[1]      
end


setcreator!(v::Variable, func::Func) = v.creator = func

# Add
mutable struct Add <: Func 
    inputs  
    outputs 
end

# 构造函数
Add() = Add(nothing, nothing) 

# 求值
_add(x1, x2) = Add()(x1, x2) do x1, x2  
    x1 .+ x2
end                            

# 为已有函数创建新方法
Base.:+(x1::Variable, x2::Variable) = _add(x1, x2) 


# main
x1 = Variable([2, 3])
x2 = Variable(3)
y = x1 + x2
print(y.value)