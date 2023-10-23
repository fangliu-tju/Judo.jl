# core

# 全局常量， 根据要不要对函数求导， 设置 `true` 或 `false`, 默认 `true`
const enable_grad = Ref(true) 

# 只做推理时不求导
macro inference(ex)
    quote
        enable_grad[] = false
        local val = Base.@__tryfinally($(esc(ex)),
        enable_grad[] = true);  
        val
    end
end

# 定义抽象数据类型， 方便代码复用
abstract type AbstractVar end

# 设置不同的变量类型
struct Variable <: AbstractVar end  # 一般变量， 主要作为目标值, 需要求导
struct Parameter <: AbstractVar end # 参数变量， 对应模型， 需要优化
struct Literal <: AbstractVar end   # 一般为常数， 或明确不需要求导的量

# 核心数据结构， 使变量具备自动微分属性
mutable struct Var{T<:AbstractVar}
    value       # 变量的取值
    grad        # 上游变量对该变量的导数值
    creator     # 该变量的创建函数
    generation  # 该变量的辈分
    name        # 该变量的名字
    
    # 内部构造函数， 覆盖默认构造函数
    function Var{T}(data::AbstractArray, name) where T   
        v = new{T}(convert.(Float64,(data)))
        v.grad = nothing
        v.creator = nothing
        v.generation = 1
        v.name = name
        return v
    end
    Var{T}(data::Number, name) where T = Var{T}([data], name)
end
# 外部构造函数， 使用循环， 方便添加变量类型
for f in (:Variable, :Parameter, :Literal)
    @eval ($f)(data; name=nothing) = Var{($f)}(data, name)
end

# 核心数据结构， 使函数具备自动微分属性
abstract type Func end  

# 通过宏创建函数类型， 实现代码复用
macro createfunc(name, arg...) 
    return quote
        mutable struct $(esc(name)) <: Func
            $(arg...)
            inputs
            outputs
            x_shape
            generation
            $(esc(name))($(arg...)) = new($(arg...),nothing,nothing,nothing,nothing)
        end
    end
end

# 实现函数求值， 及在这个过程中建立起函数和变量之间的相互联系
function (f::Func)(fun, inputs...)
    inputs = asvariable.(inputs) 
    xs = [input.value for input in inputs] 
    ys = fun(xs...)   
    isa(ys, Tuple) || (ys = tuple(ys))

    R = promote_type(typeof.(inputs)...)
    outputs = convert.(R, ys)
    
    if enable_grad[] 

        f.inputs = inputs       
        f.outputs = outputs
        f.x_shape = length(inputs) == 1 ? size(inputs[1]) : size.(inputs) #参数形状
        f.generation = mapreduce(x->x.generation, max, inputs) 
        

        for output in outputs 
            setcreator!(output, f) 
        end
    end

    return length(outputs) == 1 ? outputs[1] : outputs        
end

# 由于广播会有很多问题， 调试很麻烦， 所以这里对运算符进行了重写， 不使用广播功能， 可实现需要的功能
# Add
# 1、创建
@createfunc Add
# 2、 求值
_add(x, y) = Add()(x, y) do x, y  
    x .+ y
end                            
# 3、 扩展
Base.:+(x::Var, y::Var) = _add(x, y)
Base.:+(x, y::Var) = _add(x, y) 
Base.:+(x::Var, y) = _add(x, y) 
# 4、求导
function ∇(f::Add, gy)  
    gx1, gx2 = gy, gy
    if f.x_shape[1] != f.x_shape[2]    # 解决了广播计算， 导数维数不对的问题
        gx1 = sumto(gx1, f.x_shape[1])
        gx2 = sumto(gx2, f.x_shape[2])
    end
    return gx1, gx2    
end

# Mul
# 1、创建
@createfunc Mul
# 2、求值
_mul(x, y) = Mul()(x, y) do x, y
    x .* y
end
# 3、扩展
Base.:*(x::Var, y::Var) = _mul(x, y) 
Base.:*(x::Var, y) = _mul(x, y) 
Base.:*(x, y::Var) = _mul(x, y) 
# 4、求导
function ∇(f::Mul, gy)
    x1, x2 = f.inputs
    gx1 = gy * x2
    gx2 = gy * x1
    if f.x_shape[1] != f.x_shape[2]
        gx1 = sumto(gx1, f.x_shape[1])
        gx2 = sumto(gx2, f.x_shape[2])
    end
    return gx1, gx2
end    # 只有对 `Var` 的计算， 才能在求值时建立起联系
# Neg
# 1、创建
@createfunc Neg
# 2、求值
_neg(x) = Neg()(x) do x
    -x
end
# 3、扩展
Base.:-(x::Var) = _neg(x)
# 4、求导
∇(f::Neg, gy) = -gy

# Sub
# 1、创建
@createfunc Sub
# 2、求值
_sub(x, y) = Sub()(x, y) do x, y
    x .- y
end
# 3、扩展
Base.:-(x::Var, y::Var) = _sub(x, y)
Base.:-(x::Var, y) = _sub(x, y)
Base.:-(x, y::Var) = _sub(x, y)
# 4、求导
function ∇(f::Sub, gy) 
    gx1, gx2 = gy, -gy
    if f.x_shape[1] != f.x_shape[2]
        gx1 = sumto(gx1, f.x_shape[1])
        gx2 = sumto(gx2, f.x_shape[2])
    end
    return gx1, gx2
end

# Div
# 1、创建
@createfunc Div
# 2、求值
_div(x, y) = Div()(x, y) do x, y
    x ./ y
end
# 3、扩展
Base.:/(x::Var, y::Var) = _div(x, y)
Base.:/(x::Var, y) = _div(x, y)
Base.:/(x, y::Var) = _div(x, y)
# 4、求导
function ∇(f::Div, gy) 
    x1, x2 = f.inputs
    gx1 = gy / x2
    gx2 = gy * (-x1 / x2^2)
    if f.x_shape[1] != f.x_shape[2]
        gx1 = sumto(gx1, f.x_shape[1])
        gx2 = sumto(gx2, f.x_shape[2])
    end
    return gx1, gx2
end

# Pow
# 1、创建
@createfunc Pow c::Real
# 2、求值
_pow(x, c) = Pow(c)(x) do x
    x .^c
end
# 3、扩展
Base.:^(x::Var, c)  = _pow(x, c)
Base.literal_pow(f::typeof(^), x::Var, ::Val{c}) where c =_pow(x, c)

# 4、求导
function ∇(f::Pow, gy) 
    x = f.inputs[1]
    e = f.c
    if e == 0
        nothing
    elseif e == 1
        gy
    else
        e * x^(e - 1) * gy
    end
end

# 求整体导数
function gradient!(v::Var; retain_grad=false)
    # 非函数创建的变量，不求导
    hascreator(v) || return 
    # 将梯度转换成 `Var` 类型
    hasgrad(v) || (v.grad = Literal(one.(v.value))) 
    funcs = Func[] 
    seen_set = Set() 
    
    function addfunc(f) 
        if f ∉ seen_set       
            push!(funcs, f)   
            push!(seen_set, f)
            sort!(funcs, lt=(a, b) -> a.generation < b.generation) 
        end
    end

    addfunc(v.creator)     
     
    while !isempty(funcs)
        f = pop!(funcs)    
        gys = [output.grad for output in f.outputs] 
        #----------------------------------------------------------
        # 这部分是导数的求值过程， 配合 `@inference` 宏， 这部分基本不用修改 
        gxs = ∇(f, gys...) # 求输入参数的导数
        isa(gxs, Tuple) || (gxs = tuple(gxs)) 
        # 对导数进行分配， 然后自动配置下一步求导过程
        for (x, gx) in zip(f.inputs, gxs)
            if hasgrad(x)
                x.grad += gx
            else
                x.grad = gx
            end
            if hascreator(x)
                addfunc(x.creator)  # 中间变量
            else
                x isa Var{Literal} && cleargrad!(x) # 清除常量的梯度
            end
        end
        #-----------------------------------------------------------
        retain_grad || cleargrad!.(f.outputs)
    end
end

