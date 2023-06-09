using LinearAlgebra

function convolution(input::Matrix{Float64}, kernel::Array{Float64}, going_back::Bool=false)
    input_height, input_width = size(input)
    input_channels = 1
    kernel_height, kernel_width, kernel_channels, filters = size(kernel)
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output = zeros(output_width,output_height,filters)
    
    for filter in 1:filters
        for input_channel in 1:input_channels
            for h in 1:output_height
                for w in 1:output_width
                    for kernel_h in kernel_height
                        for kernel_w in kernel_width
                            output[h, w, filter] += kernel[kernel_height, kernel_width, input_channel, filter] * input[h-1+kernel_height, w-1+kernel_width, input_channel]
                        end
                    end
                end
            end
        end
    end
    return output
end

function convolution(input::Array{Float64}, kernel::Array{Float64}, going_back::Bool=false)
    input_height, input_width, input_channels = size(input)
    kernel_height = size(kernel,1)
    kernel_width = size(kernel,2)
    filters = last(size(kernel))
    output_height = abs(input_height - kernel_height) + 1
    output_width = abs(input_width - kernel_width) + 1
    output = zeros(output_width,output_height,filters)
    
    for filter in 1:filters
        for input_channel in 1:input_channels
            output[:,:,filter] += convolution(input[:,:,input_channel], kernel[:,:,input_channel,filter]) 
        end
    end
    return output
end
        
function convolution(input::Matrix{Float64}, kernel::Matrix{Float64}, going_back::Bool=false)
    input_height, input_width = size(input)
    input_channels = 1
    kernel_height, kernel_width = size(kernel)
    filters = 1
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output = zeros(output_width,output_height)
    
    for filter in 1:filters
        for input_channel in 1:input_channels
            for h in 1:output_height
                for w in 1:output_width
                    for kernel_h in kernel_height
                        for kernel_w in kernel_width
                            output[h, w, filter] += kernel[kernel_height, kernel_width] * input[h-1+kernel_height, w-1+kernel_width]
                        end
                    end
                end
            end
        end
    end
    return output
end
    
function convolution_backwards(input::Array{Float64}, kernel::Array{Float64}, grad::Array{Float64})
    pad_grad = zero_padding(grad,4)
    dx = convolution_back_x(pad_grad, kernel)
    dw = convolution_back_weights(input, grad)
    return dx, dw
end
    
function fully_connected(input, weight, activation=nothing)
    output = weight * input 
    if activation != nothing
        output = activation(output)
    end
    
    return output
end
    
function fully_connected_backwards(x, weight, grad)
    dx =  (grad' * weight)'
    dw = (x * grad')'
    return dx, dw

end


function convolution_back_weights(input::Array{Float64}, kernel::Array{Float64}, going_back::Bool=false)
    input_height, input_width, input_channels = size(input)
    kernel_height = size(kernel,1)
    kernel_width = size(kernel,2)
    filters = size(kernel,3)
    output_height = abs(input_height - kernel_height) + 1
    output_width = abs(input_width - kernel_width) + 1
    output = zeros(output_width,output_height, input_channels, filters)
    
    for filter in 1:filters
        for input_channel in 1:input_channels
            #@show convolution2(input[:,:,input_channel],kernel[:,:, filter])
            output[:,:,input_channel,filter] += convolution_back_matrix(input[:,:,input_channel],kernel[:,:, filter]) 
        end
    end
    return output
end
function convolution_back_weights(input::Matrix{Float64}, kernel::Array{Float64}, going_back::Bool=false)
    input_height, input_width = size(input)
    input_channels = 1
    kernel_height = size(kernel,1)
    kernel_width = size(kernel,2)
    filters = size(kernel,3)
    output_height = abs(input_height - kernel_height) + 1
    output_width = abs(input_width - kernel_width) + 1
    output = zeros(output_width,output_height, input_channels, filters)
    
    for filter in 1:filters
        for input_channel in 1:input_channels
            for h in 1:output_height
                for w in 1:output_width
                    for kernel_h in 1:kernel_height
                        for kernel_w in 1:kernel_width
                            output[h, w, input_channel,filter] += kernel[kernel_h, kernel_w, filter] * input[h-1+kernel_h, w-1+kernel_w]
                        end
                    end
                end
            end
        end
    end
    return output
end
function convolution_back_x(input::Array{Float64}, kernel::Array{Float64}, going_back::Bool=false)
    input_height, input_width, input_channels = size(input)
    kernel_height = size(kernel,1)
    kernel_width = size(kernel,2)
    filters = size(kernel,3)
    output_height = abs(input_height - kernel_height) + 1
    output_width = abs(input_width - kernel_width) + 1
    output = zeros(output_width,output_height, filters)
    
    for filter in 1:filters
        for input_channel in 1:input_channels
            #@show convolution2(input[:,:,filter],kernel[:,:, filter, input_channel])
            output[:,:,filter] += convolution_back_matrix(input[:,:,filter],kernel[:,:, filter, input_channel]) 
        end
    end
    return output
end
function convolution_back_matrix(input::Matrix{Float64}, kernel::Matrix{Float64}, going_back::Bool=false)
    input_height, input_width = size(input)
    input_channels = 1
    kernel_height, kernel_width = size(kernel)
    filters = 1
    output_height = abs(input_height - kernel_height) + 1
    output_width = abs(input_width - kernel_width) + 1
    output = zeros(output_width,output_height)
    
    for filter in 1:filters
        for input_channel in 1:input_channels
            for h in 1:output_height
                for w in 1:output_width
                    for kernel_h in 1:kernel_height
                        for kernel_w in 1:kernel_width
                            output[h, w, filter] += kernel[kernel_h, kernel_w] * input[h-1+kernel_h, w-1+kernel_w]
                        end
                    end
                end
            end
        end
    end
    return output
end

function relu(x::Union{Matrix, Vector, Array, Int, Transpose{Float64, Vector{Float64}}})
    return max.(0, x)
end

function relu_backwards(x::Union{Matrix, Vector, Array, Int, Transpose{Float64, Vector{Float64}}})
    y = copy(x)
    y[y .<= 0] .= 0
    y[y .> 0] .= 1
    return y
end

function softmax(x::Union{Matrix, Vector, Array, Int, LinearAlgebra.Transpose{Float64, Vector{Float64}}}) 
    exp_x = exp.(x) 
    return exp_x ./ sum(exp_x)
end

function softmax_backwards(output)
    if size(output,2) == 1
        softmax_grad = diagm(output) - output .* output'
    else
        softmax_grad = zeros(size(output))
        for i in 1:size(output, 1)
            for j in 1:size(output, 2)
                if j > 1
                    if i == j
                        softmax_grad[i, j] = output[i, j] * (1 - output[i, j])
                    else
                        softmax_grad[i, j] = -output[i, j] * output[j, i]
                    end
                else
                    if i == j
                        softmax_grad[i, j] = output[i] * (1 - output[i, j])
                    else
                        softmax_grad[i, j] = -output[i] * output[j, i]
                    end
                end
            end
        end
    end
    return softmax_grad
end

function pooling(x::Union{Matrix, Vector, Array, Int}, kernel::Int=2, stride::Int=kernel, pool_type::String="max")
    channels = size(x,3)
    output_shape = (size(x,1) - kernel) / stride + 1
    output_shape = trunc(Int, output_shape)
    output = zeros((output_shape, output_shape, channels))
    for channel in 1:channels
        for i in 1:stride:size(x,1)
            for j in 1:stride:size(x,2)
                if pool_type == "max"
                    output[fld(i,stride)+1, fld(j,stride)+1, channel] = maximum(x[i:i+kernel-1,j:j+kernel-1,channel])
                elseif pool_type == "mean"
                    output[i/stride + 1, j/stride + 1, channel] = mean(x[i:i+kernel-1,j:j+kernel-1, channel])
                end
            end
        end
    end      
    return output
end

function pooling_backwards(x::Union{Matrix, Vector, Array, Int}, output::Union{Matrix, Vector, Array, Int}, grad::Union{Matrix, Vector, Array, Int}, kernel::Int=2, stride::Int=kernel, pool_type::String="max")    
     channels = size(x,3)
     output_shape = size(x,1)
     pool_grad = zeros(size(x))
     output_flat = vec(output)
     kernel = stride = 2
     for channel in 1:channels
        for i in 1:stride:size(x,1)
            for j in 1:stride:size(x,2)
                local_max, max_location = findmax(x[i:i+kernel-1,j:j+kernel-1, channel])
                pool_grad[max_location[1]+i-1,max_location[2]+j-1,channel] = grad[fld(i,stride)+1,fld(j,stride)+1,channel]
            end
        end
    end
    return pool_grad     
end

function zero_padding(x::Array{Float64}, pad)
    (height, width, channels) = size(x)

    out_height = height + 2*pad
    out_width = width + 2*pad

    out = zeros(Float64, out_height, out_width, channels)
    
    out[pad+1:end-pad, pad+1:end-pad, :] = x

    return out
end

using Plots

function display_image(image::Array{Float64,2})
    plot(
        heatmap(image,
                aspect_ratio=:equal,
                frame=:none,
                c=:grays,
                colorbar=false,
                legend=false),
        ticks=nothing,
        border=:none,
        axis=:off
    )
end

function int_to_array(x::Int64, length::Int64)
    output = zeros(length)
    output[x+1] = 1
    return output'
end

function tensor_product(x::AbstractArray{T,1}, y::AbstractArray{U,1}) where {T<:Real, U<:Real}
    m, n = length(x), length(y)
    z = zeros(T, m, n)
    for i in 1:m
        for j in 1:n
            z[i,j] = x[i] * y[j]
        end
    end
    return z
end

function cross_entropy_loss(pred::Vector{Float64}, dest::Int64)
    num_of_classes = size(pred,1)
    label = int_to_array(dest, num_of_classes)
    for i in 1:length(pred) if pred[i]==0 pred[i] += 1e-7 end end
    loss = -sum((label * log.(pred)))/num_of_classes
    return loss
end

function cross_entropy_loss_backwards(pred::Vector{Float64}, dest::Int64)
    num_of_classes = size(pred,1)
    label = int_to_array(dest, num_of_classes)
    return pred - label'
end

function xavier_init(in_channels::Int64, out_channels::Int64, kernel_size::Int64)
    fan_in = in_channels * kernel_size^2
    fan_out = out_channels * kernel_size^2
    variance = 2 / (fan_in + fan_out)
    return randn(Float64, kernel_size, kernel_size, in_channels, out_channels) .* sqrt(variance)
end

function xavier_init(out_channels::Int64, in_channels::Int64) 
    variance = 2 / (in_channels + out_channels)
    return randn(Float64, out_channels, in_channels) .* sqrt(variance)
end

mutable struct DenseLayer
    in_channels::Int64
    out_channels::Int64
    weights::Array
    activation::Function
    make_flat::Bool
end

function DenseLayer(in_channels::Int64, out_channels::Int64, activation::Function=nothing, make_flat::Bool=false)
    std_dev = sqrt(2/(in_channels*out_channels*out_channels))
    weights1 = randn(Float32, out_channels, in_channels) .* std_dev 
    weights = xavier_init(out_channels, in_channels)
    return DenseLayer(in_channels, out_channels, weights, activation, make_flat)
end

mutable struct ConvLayer
    in_channels::Int64
    out_channels::Int64
    weights::Array
    kernel_size::Int64
    activation::Function
end

function ConvLayer(in_channels::Int64, out_channels::Int64, kernel_size::Int64, activation::Function=nothing)
    std_dev = sqrt(2/(in_channels*out_channels*out_channels))
    weights1 = randn(Float64, kernel_size, kernel_size, in_channels, out_channels) .* std_dev
    weights = xavier_init(in_channels, out_channels, kernel_size)
    return ConvLayer(in_channels, out_channels, weights, kernel_size, activation)
end

struct LeNet5
    conv1::ConvLayer
    conv2::ConvLayer
    fc1::DenseLayer
    fc2::DenseLayer
    fc3::DenseLayer
end

function intialize_network()
    conv1 = ConvLayer(1, 6, 5, relu)
    conv2 = ConvLayer(6, 16, 5, relu)
    fc1 = DenseLayer(16*5*5, 120, relu, true)
    fc2 = DenseLayer(120, 84, relu)
    fc3 = DenseLayer(84, 10, softmax)  
    # Unlike in reference_solution.py, here the softmax function is done at the end of third dense layer instead of 
    # at the beginning of cross entropy loss calculation, operation wise it should make no difference 
    return LeNet5(conv1, conv2, fc1, fc2, fc3)
end


using MLDatasets, Images, ImageMagick, Shuffle

function load_train_data(shape::Tuple=(32,32), batch_size::Int=64)
    images, labels = MNIST.traindata(Float64)
    resized_images = []
    for i in 1:size(images)[3]
        img = images[:,:,i]
        resized_img = imresize(img, (32, 32))
        push!(resized_images, resized_img)
    end
    
    num_samples = length(resized_images)
    num_batches = div(num_samples, batch_size)
    batches = []
    indices = shuffle(1:num_samples)
    
    for i in 1:num_batches
        batch_indices = indices[(i-1)*batch_size + 1:i*batch_size]
        batch_images = [resized_images[j] for j in batch_indices]
        batch_labels = [labels[j] for j in batch_indices]
        push!(batches, (batch_images, batch_labels))
    end

    return batches
end
    
function load_test_data(shape::Tuple=(32,32), batch_size::Int=64)
    images, labels = MNIST.testdata(Float64)
    resized_images = []
    for i in 1:size(images)[3]
        img = images[:,:,i]
        resized_img = imresize(img, (32, 32))
        push!(resized_images, resized_img)
    end
    
    num_samples = length(resized_images)
    num_batches = div(num_samples, batch_size)
    batches = []
    indices = shuffle(1:num_samples)
    
    for i in 1:num_batches
        batch_indices = indices[(i-1)*batch_size + 1:i*batch_size]
        batch_images = [resized_images[j] for j in batch_indices]
        batch_labels = [labels[k] for k in batch_indices]
        push!(batches, (batch_images, batch_labels))
    end
    
    return batches
end


function forward(net::LeNet5, input::Union{Matrix{Float64},Vector})
    forward_results = []
    x = convolution(input, net.conv1.weights); push!(forward_results,x)
    x = net.conv1.activation(x); push!(forward_results,x)    
    x = pooling(x); push!(forward_results,x)
    x = convolution(x, net.conv2.weights); push!(forward_results,x)
    x = net.conv2.activation(x); push!(forward_results,x)
    x = pooling(x); push!(forward_results,x)
    x = collect(Iterators.flatten(x))
    x = fully_connected(x, net.fc1.weights); push!(forward_results,x)
    x = net.fc1.activation(x); push!(forward_results,x)
    x = fully_connected(x, net.fc2.weights); push!(forward_results,x)
    x = net.fc2.activation(x); push!(forward_results,x)
    x = fully_connected(x, net.fc3.weights); push!(forward_results,x)
    x = net.fc3.activation(x); push!(forward_results,x)
    return forward_results
end

function backward(model::LeNet5, image::Array, preds::Vector, loss::Number, dest::Int64, learning_rate::Float64=0.0016)
    
    # zero grad
    grad_c1 = zeros(size(model.conv1.weights))
    grad_c2 = zeros(size(model.conv2.weights))
    grad_fc1 = zeros(size(model.fc1.weights))
    grad_fc2 = zeros(size(model.fc2.weights))
    grad_fc3 = zeros(size(model.fc3.weights))

    # Create graph and calculate grad for loss function
    graph = copy(preds); push!(graph, loss)
    # gradients = fill(nothing, (1,length(preds))); push!(gradients, cross_entropy_loss_backwards(last(preds),dest))
    curr_grad = cross_entropy_loss_backwards(preds[end-1],dest)
    sequence = ["conv1", net.conv1.activation,pooling,"conv2",net.conv2.activation,pooling,"deflatten","fc1",net.fc1.activation,"fc2",net.fc2.activation,"fc3",net.fc3.activation]
    # Backpropagate through each layer
    for i = length(graph):-1:1
        input = graph[i]
        if sequence[i] == softmax
            curr_grad = softmax_backwards(curr_grad) * curr_grad
        elseif sequence[i] == relu
            curr_grad = relu_backwards(graph[i-1]) .* curr_grad
        elseif sequence[i] == pooling
            curr_grad = pooling_backwards(graph[i-1],graph[i],curr_grad)
        elseif sequence[i] == "fc3"
            curr_grad, grad_fc3 = fully_connected_backwards(graph[i-2],model.fc3.weights,curr_grad)
        elseif sequence[i] == "fc2"
            curr_grad, grad_fc2 = fully_connected_backwards(graph[i-2],model.fc2.weights,curr_grad)
        elseif sequence[i] == "fc1"
            curr_grad, grad_fc1 = fully_connected_backwards(collect(Iterators.flatten(graph[i-2])),model.fc1.weights,curr_grad)
        elseif sequence[i] == "conv2"
            curr_grad, grad_c2 = convolution_backwards(graph[i-1],model.conv2.weights,curr_grad)
        elseif sequence[i] == "conv1"
            curr_grad, grad_c1 = convolution_backwards(image,model.conv1.weights,curr_grad)
        elseif sequence[i] == "deflatten"
            curr_grad = reshape(curr_grad,5,5,16)
        end
    end
    
    # Update model parameters
    model.conv1.weights += learning_rate .* grad_c1
    model.conv2.weights += learning_rate .* grad_c2
    model.fc1.weights += learning_rate .* grad_fc1
    model.fc2.weights += learning_rate .* grad_fc2
    model.fc3.weights += learning_rate .* grad_fc3
    return loss, model
#     return grad_c1, grad_c2, grad_fc1, grad_fc2, grad_fc3
end

using Statistics
""" 
    Training that takes uses every individual image and label for backpropagation
"""

function training_naive(net::LeNet5, train_data::Vector, criterion::Function, num_of_epochs::Int64=10, batches_per_epoch::Int64=50, batch_size::Int64=64)
    # Forward and backward inference example
    loss_report = fld(batches_per_epoch, 2)  # Report on loss twice per epoch
    for epoch in 1:num_of_epochs
        
        losses_in_epoch = []
        
        for batch in 1:batches_per_epoch
            
            preds_stack = []
            
            for observation in 1:batch_size
                
                image = train_data[batch][1][observation]
                dest = train_data[batch][2][observation]

                preds = forward(net, image)
                last(preds)
                loss = criterion(last(preds), dest)
                push!(losses_in_epoch,loss)
                loss1, net = backward(net,image,preds,loss,dest,0.016)

            end

            if batch%loss_report == 0
                println("LOSS: ", mean(losses_in_epoch), " EPOCH: ", epoch)
            end

        end
    end
end



using Statistics

""" 
Training that includes batches 
WIP

"""

function training(net::LeNet5, train_data::Vector, criterion::Function, num_of_epochs::Int64=10, batches_per_epoch::Int64=50, batch_size::Int64=64)
    loss_report = fld(batches_per_epoch, 2)  # Report on loss twice per epoch
    for k in 1:num_of_epochs
        
    losses_in_epoch = []
        
    for i in 1:batches_per_epoch
            
        preds_stack = []
            
        for j in 1:batch_size
                
            image = train_data[i][1][j]
            dest = train_data[i][2][j]

            preds = forward(net, image)
            push!(preds_stack,preds)
            loss = criterion(last(preds), dest)
            push!(losses_in_epoch,loss)
        end
            
        loss_m = mean(losses_in_epoch)
            
        for l in 1:64
            image1 = train_data[i][1][l]
            dest1 = train_data[i][2][l]
            preds1 = preds_stack[l]
            loss1, net = backward(net,image_1,preds1,loss_m,dest1)
        end

        if batch%loss_report == 0
             @show mean(loss2), k
        end
    end
    end
end

function testing(net::LeNet5, test_data::Vector, batch_size::Int64=64)
    correct = 0
    data_length = length(test_data)
    acc_report = fld(data_length, 10)  # Report on loss ten times in set
    accuracy_stack = []
    for test_batch in 1:data_length
        for observation in 1:batch_size
            
            image = test_data[test_batch][1][observation]
            dest = test_data[test_batch][2][observation]
            preds = forward(net, image)
            if findmax(last(preds))[2]-1 == dest
                correct += 1
            end
        end 
        
    if test_batch%acc_report ==0
        accuracy = correct/(test_batch*64)
        println("++++ ON: ", test_batch*64, " IMAGES ++++ ACC: ", round(accuracy*100,digits=2), "% ++++")
    end
        
    end
    println("++++ FINAL ACC: ", round(correct/(data_length*64) * 100, digits=2), "% ++++" )
end

net = intialize_network()
train_data = load_train_data()
test_data = load_test_data()

training_naive(net,train_data,cross_entropy_loss)
testing(net,test_data)


