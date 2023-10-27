# ==================================================================
# Core algorithm
# ==================================================================
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

# 实现函数求值， 及在这个过程中建立起函数和变量之间的相互联系, 正向传播
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

# 求整体导数， 逆向传播
function gradient!(v::Var; retain_grad=false)
    # 非函数创建的变量，不求导
    hascreator(v) || return 
    # 将梯度转换成 `Var` 类型
    hasgrad(v) || (v.grad = Literal(ones(size(v)))) 
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

# ==================================================================
# Core functions: Add / Mul / Neg / Sub / Div / Pow
# ==================================================================
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

# ==================================================================
# Basic functions: Sin / Cos / Tanh / Exp / Log
# ==================================================================

# Sin
# 1、创建
@createfunc Sin
# 2、求值+3、扩展
Base.sin(x::Var) = Sin()(x) do x
    sin.(x)
end
# 4、求导
∇(f::Sin, gy) = gy * cos(f.inputs[1])

# Cos
# 1、创建
@createfunc Cos
# 2、求值+3、扩展
Base.cos(x::Var) = Cos()(x) do x
    cos.(x)
end
# 4、求导
∇(f::Cos, gy) = gy * -sin(f.inputs[1])

# Tanh
# 1、创建
@createfunc Tanh
# 2、求值+3、扩展
Base.tanh(x::Var) = Tanh()(x) do x
    tanh.(x)
end
# 4、求导
∇(f::Tanh, gy) = gy * (1 - f.outputs[1]^2)

# Exp
# 1、创建
@createfunc Exp
# 2、求值+3、扩展
Base.exp(x::Var) = Exp()(x) do x
    exp.(x)
end
# 4、求导
∇(f::Exp, gy) = gy * f.outputs[1]

# Log
# 1、创建
@createfunc Log
# 2、求值+3、扩展
Base.log(x::Var) = Log()(x) do x
    log.(x)
end
# 4、求导
∇(f::Log, gy) = gy / f.inputs[1]

# ===================================================================
# Tensor operations: reshape / permutedims / ajoint / getindex / getindexgrad
# ===================================================================

# Reshape
# 1、创建
@createfunc Reshape shape::Tuple
# 2、求值+3、扩展
Base.reshape(x::Var, shape::Tuple) = size(x) == shape ? x : Reshape(shape)(x) do x
    reshape(x, shape)
end
function Base.reshape(x::Var, shape...) 
    if length(shape) == 1 && shape[1] isa Union{Tuple,Array}
        shape = shape[1]
    end
    return reshape(x,tuple(shape...))
end
# 4、求导
∇(f::Reshape, gy) = reshape(gy, f.x_shape)

# Permutedims
# 1、创建
@createfunc Permutedims shape::Tuple
# 2、求值+3、扩展
Base.permutedims(x::Var, shape::Tuple) = Permutedims(shape)(x) do x
    permutedims(x, shape)
end
Base.permutedims(x::Var,shape::AbstractArray) = permutedims(x, Tuple(shape))
# 4、求导
∇(f::Permutedims, gy) = permutedims(gy, invperm(f.shape))

# Adjoint
# 1、创建
@createfunc Adjoint
# 2、求值+3、扩展
Base.adjoint(x::Var) = Adjoint()(x) do x
    adjoint(x)
end
# 4、求导
∇(f::Adjoint, gy) = adjoint(gy) 

# GetIndex
# 1、创建
@createfunc GetIndex slices 
# 2、求值+3、扩展
Base.getindex(x::Var, slices...) = GetIndex(slices)(x) do x 
    getindex(x, slices...)
end
# 4、 求导
∇(f::GetIndex, gy) = begin
    slices = f.slices
    in_shape = f.x_shape
    getindexgrad(gy, slices, in_shape)
end

# GetIndexGrad
# 1、创建
@createfunc GetIndexGrad slices in_shape 
# 2、求值
getindexgrad(gy, slices, in_shape) = GetIndexGrad(slices, in_shape)(gy) do gy 
    gx = zeros(in_shape...)
    gx[slices...] += gy
    return gx 
end
# 4、求导
∇(f::GetIndexGrad, gx) = gx[f.slices...]


# ===================================================================
# isless / sum / sumto / broadcastto / matmul / affine
# ===================================================================
# Isless
# 1、创建
@createfunc Isless
# 2、 求值
_isless(x, y) = Isless()(x, y) do x, y  
    isless.(x, y)
end                            
# 3、 扩展
Base.:isless(x::Var, y::Var) = _isless(x, y)
Base.:isless(x, y::Var) = _isless(x, y) 
Base.:isless(x::Var, y) = _isless(x, y) 
# 4、求导
function ∇(f::Isless, gy)  
    gx1, gx2 = gy, gy
    if f.x_shape[1] != f.x_shape[2]   
        gx1 = sumto(gx1, f.x_shape[1])
        gx2 = sumto(gx2, f.x_shape[2])
    end
    return gx1, gx2    
end

# Sum
# 1、创建 
@createfunc Sum dims
# 2、求值+3、扩展
Base.sum(x::Var; dims=:) = Sum(dims)(x) do x
    sum(x, dims=dims)
end
# 4、求导
∇(f::Sum, gy) = broadcastto(gy, f.x_shape)

# Sumto
# 1、创建
@createfunc SumTo shape 
# 2、求值+3、扩展
sumto(x::Var, shape) = SumTo(shape)(x) do x
    size(x) == shape && return x
    lead = ndims(x) - length(shape)
    dims = Tuple(1:lead)
    for i = 1:length(shape)
        if shape[i] == 1 && size(x, i) > 1
            dims = tuple(dims..., i)
        end
    end
    return sum(x, dims=dims)
end
# 4、求导
∇(f::SumTo, gy) = broadcastto(gy, f.x_shape)

# BroadcastTo
# 1、创建
@createfunc BroadcastTo shape 
# 2、求值+3、扩展
broadcastto(x::Var, shape) = BroadcastTo(shape)(x) do x
    size(x) == shape ? x : x .+ zeros(shape)
end
∇(f::BroadcastTo, gy) = sumto(gy, f.x_shape)

# MatMul
# 1、创建 
@createfunc MatMul
# 2、求值+3、扩展
matmul(W, x) = MatMul()(W, x) do W, x
    W = length(W) == 1 ? W[1] : W # 当数值只有一个元素时， 取出该值
    x = length(x) == 1 ? x[1] : x # 这样可以防止向量乘以向量的错误
    W * x
end
⋅(A::Var, B::Var) = matmul(A, B)
⋅(A::Var, B) = matmul(A, B)
⋅(A, B::Var) = matmul(A, B)
# 4、求导
function ∇(f::MatMul, gy)
    W, x = f.inputs
    gW = gy ⋅ x'
    gx = W' ⋅ gy
    return gW, gx
end

# Affine
# 1、创建
@createfunc Affine 
# 2、求值+3、扩展
affine(x, W, b=nothing) = Affine()(x, W, b) do x, W, b
    x = length(x) == 1 ? x[1] : x # 当数值只有一个元素时， 取出该值
    W = length(W) == 1 ? W[1] : W # 这样可以防止向量乘以向量的错误

    t = x * W
    isnothing(b) && return t
    t .+ b
end
# 4、求导
function ∇(f::Affine, gy)
    x, W, b = f.inputs
    gb = isnothing(b.value) ? nothing : sumto(gy, size(b))
    gx = gy ⋅ W'
    gW = x' ⋅ gy
    return gx, gW, gb
end


# ===================================================================
# loss funtion mean_squared_error / 
# ===================================================================

# MeanSquaredError
# 1、创建
@createfunc MeanSquaredError
# 2、求值+3、扩展
mean_squared_error(x1, x2) = MeanSquaredError()(x1, x2) do x1, x2
    diff = x1 - x2
    sum(diff.^2) / length(diff)
end 
# 4、求导
function ∇(f::MeanSquaredError, gy)
    x1, x2 = f.inputs
    diff = x1 - x2
    gx1 = gy * diff * (2 / length(diff))
    gx2 = -gx1
    return gx1, gx2
end

# ===================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# ===================================================================

# Sigmoid
# 1、创建
@createfunc Sigmoid 
# 2、求值+3、扩展
sigmoid(x) = Sigmoid()(x) do x 
    0.5tanh.(0.5x) .+ 0.5
end
# 4、求导
function ∇(f::Sigmoid, gy) 
    y = f.outputs[1]
    gy * y * (1 - y)
end

# ReLU
# 1、创建
@createfunc ReLU 
# 2、求值+3、扩展
relu(x) = ReLU()(x) do x
    max.(x, 0.0)
end
# 4、求导
function ∇(f::ReLU, gy)
    x = f.inputs[1]
    mask = x > 0
    gy * mask
end

# Softmax
# 1、创建
@createfunc Softmax dims 
# 2、求值+3、扩展
softmax(x; dims=2) = Softmax(dims)(x) do x 
    y = x .- maximum(x, dims=dims)
    y = exp.(y)
    y ./ sum(y, dims=dims)
end
# 4、求导
function ∇(f::Softmax, gy) # 还没有定义完！！！
    y = f.outputs[1]
end

#=


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


class LogSoftmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)


class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        gx = gy * mask
        return gx


def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)
=#

# Max
@createfunc Max
backward(f::Max, gy) = begin #该函数的实现是不合理的，后面需要改进
    x1, x2 = f.inputs
    y = f.outputs[1]
    ET = eltype(gy)
    T = typeof(gy)
    gx1 = T(ET.(x1.data .== y.data))
    gx2 = T(ET.(x2.data .== y.data))
    if f.x_shape[1] != f.x_shape[2]    # 当维数不一样时，需要累加运算
        gx1 = sumto(gx1, f.x_shape[1])
        gx2 = sumto(gx2, f.x_shape[2])
    end

    return gx1, gx2
end
varmax(x1, x2) = Max()(x1, x2) do x1, x2
    max.(x1.data, x2.data)
end
Base.max(x::Variable, y::Variable) = varmax(x, y)
Base.max(x::Variable, y) = varmax(x, y)
Base.max(x, y::Variable) = varmax(x, y)