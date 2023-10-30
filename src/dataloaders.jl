# dataloaders

# 定义抽象数据集类型
abstract type DataLoader end

# 创建数据集类型
macro createloader(name)
    return quote
        struct $(esc(name)) <: DataLoader
            dataset
            batch_size
            shuffle
            gpu
            data_size
            max_iter
            index
            function $(esc(name))(dataset, batch_size;shuffle=true, gpu=false)
                data_size = size(dataset,1)
                max_iter = data_size ÷ batch_size 
                index = if shuffle
                    randperm(data_size)
                else
                    collect(1:data_size)
                end

                new(dataset,batch_size,shuffle,gpu,data_size,max_iter, index)
            end
        end
    end
end

@createloader RndDataLoader
