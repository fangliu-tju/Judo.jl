# step11

mutable struct Variable
    value
    grad
    creator
end
Variable(data) = Variable(data, nothing, nothing)
Variable() = Variable(nothing, nothing, nothing)

abstract type Func end  
function (f::Func)(fun::Function, inputs::Vector{Variable}) # 输入是向量， 暂时的
    xs = [input.value for input in inputs] # 遍历数组      
    ys = fun(xs)   # 这步中用的函数是针对数组运算的 `sum`
    outputs = Variable.(ys) # `Variable` 是针对单个参数的， 对于数组要用到 `.` 运算符
    for output in outputs          
        setcreator!(output, f) # 每个输出都使用了相同的运算
    end         
    f.inputs = inputs       
    f.outputs = outputs
    return outputs      
end


setcreator!(v::Variable, func::Func) = v.creator = func

# Sum
mutable struct Sum <: Func 
    inputs  
    outputs 
end

# 构造函数
Sum() = Sum(nothing, nothing) 

# 求值
_sum(xs) = Sum()(xs) do xs  
    y = sum(xs)
    return [y]    
end                            

# 为已有函数创建新方法
Base.sum(xs::Vector{Variable}) = _sum(xs) 


# main
xs = [Variable([2.0]), Variable([3.0])]
ys = sum(xs)
y = ys[1]
print(y.value)

#这步的实现是一个过渡， 所以有些别扭， 下一步将解决这个问题。