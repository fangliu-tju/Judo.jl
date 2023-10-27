# variables

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
