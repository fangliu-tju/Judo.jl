# Optimizer

# 新定义一个数据类型， 优化器
abstract type Optimizer end

# 创建优化器的宏
macro createopt(name, args...)
    return quote
        mutable struct $(esc(name)) <: Optimizer
            $(args...)
            target
            hooks
            $(esc(name))($(args...)) = new($(args...), nothing, Function[])
        end
    end
end

setup!(opt::Optimizer, target) = opt.target = target

function update!(opt::Optimizer)
    ps = [p for p in params(opt.target) if !isnothing(p.grad)]

    for f in opt.hooks
        f(params)
    end

    for p in ps
        update_one!(opt, p)
    end
end

add_hook(opt, f) = append!(opt.hooks, f)


# SGD
# 1、创建优化器
@createopt SGD lr
SGD(;lr=0.01) = SGD(lr)
# 2、定义优化的更新算法
function update_one!(opt::SGD, param)
    param.value -= opt.lr * param.grad.value
end

# MomentumSGD
# 1、创建优化器
@createopt MomentumSGD lr momentum vs
MomentumSGD(;lr=0.01,momentum=0.9,vs=Dict()) = MomentumSGD(lr, momentum, vs)
# 2、定义优化的更新算法
function update_one!(opt::MomentumSGD, p)
    v_key = objectid(p) # 获得参数的唯一标识
    v = get!(opt.vs, v_key, zero(p.value)) # 获得上一步骤的速度
    v *= opt.momentum
    v -= opt.lr * p.grad.value
    opt.vs[v_key] = v # 更新速度
    p.value += v      # 更新值
end