# ==================================================================
# Basic functions: sin / cos / tanh / exp / log
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
# Tensor operations: reshape / permutedims / get_item / expand_dims / flatten
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


# ===================================================================
# sum / average / matmul / linear /max
# ===================================================================

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
    size(x) == shape ? x : x .+ zeros(eltype(x), shape)
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

#=
class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


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