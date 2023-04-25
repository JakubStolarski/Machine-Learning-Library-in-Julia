using LinearAlgebra

function convolution(input::Matrix{Float64}, kernel, activation=nothing)
    input_height, input_width = size(input)
    if length(size(kernel)) == 4
        kernel_height, kernel_width, in_channels, out_channels = size(kernel)
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        output = zeros(output_height, output_width, out_channels)

        for in_channel in 1:in_channels
            for out_channel in 1:out_channels
                for i in 1:output_height
                    for j in 1:output_width
                        for k in 1:kernel_height
                            for l in 1:kernel_width
                                output[i, j, out_channel] += input[i+k-1, j+l-1] * kernel[k, l, in_channel, out_channel]
                            end
                        end
                    end
                end
            end
        end
    elseif length(size(kernel)) == 3
        kernel_height, kernel_width, out_channels = size(kernel)
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        output = zeros(output_height, output_width, out_channels)
    
                for out_channel in 1:out_channels
                    for i in 1:output_height
                        for j in 1:output_width
                            for k in 1:kernel_height
                                for l in 1:kernel_width
                                    output[i, j, out_channel] += input[i+k-1, j+l-1] * kernel[k, l, out_channel]
                                end
                            end
                        end
                    end

        end
    elseif length(size(kernel)) == 2
        kernel_height, kernel_width = size(kernel)
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        output = zeros(output_height, output_width)

                for i in 1:output_height
                    for j in 1:output_width
                        for k in 1:kernel_height
                            for l in 1:kernel_width
                                output[i, j] += input[i+k-1, j+l-1] * kernel[k, l]
                            end
                        end
                    end
                end
        
    end
    if activation != nothing
        output = activation(output)
    end
    return output
end

function convolution(input::Array{Float64}, kernel, activation=nothing)
    input_height, input_width, input_channels = size(input)
    if length(size(kernel)) == 4 
        kernel_height, kernel_width, in_channels, out_channels = size(kernel)
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        output = zeros(output_height, output_width, out_channels)

        for input_channel in 1:input_channels
            for in_channel in 1:in_channels
                for out_channel in 1:out_channels
                    for i in 1:output_height
                        for j in 1:output_width
                            for k in 1:kernel_height
                                for l in 1:kernel_width
                                    output[i, j, out_channel] += input[i+k-1, j+l-1, input_channel] * kernel[k, l, in_channel, out_channel]
                                end
                            end
                        end
                    end
                end
            end
        end
    else
        kernel_height, kernel_width, out_channels = size(kernel)
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        output = zeros(output_height, output_width, out_channels)

        for input_channel in 1:input_channels
            
                for out_channel in 1:out_channels
                    for i in 1:output_height
                        for j in 1:output_width
                            for k in 1:kernel_height
                                for l in 1:kernel_width
                                    output[i, j, out_channel] += input[i+k-1, j+l-1, input_channel] * kernel[k, l, out_channel]
                                end
                            end
                        end
                    end
                
            end
        end
    end
    if activation != nothing
        output = activation(output)
    end
    
    return output
end

function convolution_backwards(input::Array{Float64}, kernel::Array{Float64}, grad::Array{Float64})
    dw = convolution(grad,input)
    pad = size(input,1) - size(grad,1)
    grad_pad = zero_padding(grad, pad)
    dx = convolution(grad_pad, rot180(kernel))
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
    
    dx = weight' * grad
    dw = grad * x'
    return dx, dw

end

function relu(x::Union{Matrix, Vector, Array, Int, Transpose{Float64, Vector{Float64}}})
    return max.(0, x)
end

function relu_backwards(x::Union{Matrix, Vector, Array, Int, Transpose{Float64, Vector{Float64}}})
    x[x .<= 0] .= 0
    x[x .> 0] .= 1
    return x
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

mutable struct DenseLayer
    in_channels::Int64
    out_channels::Int64
    weights::Array
    activation::Function
    make_flat::Bool
end

function DenseLayer(in_channels::Int64, out_channels::Int64, activation::Function=nothing, make_flat::Bool=false)
    std_dev = sqrt(2/(in_channels*out_channels*out_channels))
    weights = randn(Float32, out_channels, in_channels) .* std_dev 
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
    weights = randn(Float64, kernel_size, kernel_size, in_channels, out_channels) .* std_dev 
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
    
    num_samples = length(images_resized)
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

function backward(model::LeNet5, image::Array, preds::Vector, loss::Number, dest::Int64, learning_rate::Float64=0.16)
    
    # zero grad
    grad_c1 = zeros(size(model.conv1.weights))
    grad_c2 = zeros(size(model.conv2.weights))
    grad_fc1 = zeros(size(model.fc1.weights))
    grad_fc2 = zeros(size(model.fc2.weights))
    grad_fc3 = zeros(size(model.fc3.weights))

    # Create graph and calculate grad for loss function
    graph = copy(preds); push!(graph, loss)
    # gradients = fill(nothing, (1,length(preds))); push!(gradients, cross_entropy_loss_backwards(last(preds),dest))
    curr_grad = cross_entropy_loss_backwards(last(preds),dest)
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
    model.conv1.weights[:,:,1,:] -= learning_rate .* grad_c1
    for i in 1:6
    model.conv2.weights[:,:,i,:] -= learning_rate .* grad_c2
    end
    model.fc1.weights -= learning_rate .* grad_fc1
    model.fc2.weights -= learning_rate .* grad_fc2
    model.fc3.weights -= learning_rate .* grad_fc3
    return loss, model
end

function training(model::LeNet5, train_data::Vector, criterion::Function, num_of_epochs::Int64=10, batch_size::Int64=64)
    num_of_steps = length(train_data)
    for epoch in 1:num_of_epochs
        j = 1
        for batch in 1:num_of_steps
            images = train_data[batch][1]
            labels = train_data[batch][2]
            gradients = []
            i = 1
            for(image,label) in zip(images,labels)
                outputs = forward(model, image)
                loss = criterion(last(outputs), label)
                loss, model = backward(model, image, outputs, loss, label)
                # push!(loss, criterion(last(outputs), label))
                i = i + 1
                println("loss:", loss, "num:", i)
            end
            println("Batch:", loss, "num:", j)
            j = j+1
        end
        
       
    end
    end


# Forward and backward inference example
train_data = load_train_data()
net = intialize_network()

# u = train_data[1][1][1]
# dest = train_data[1][2][1]
# preds = forward(net, u)
# last(preds)
# loss = cross_entropy_loss(last(preds), 1)
# loss, net = backward(net,u,preds,loss,dest) 
# # println(loss)

# @time u = train_data[1][1][2]
# @time dest = train_data[1][2][2]
# @time preds = forward(net, u)
# @time loss = cross_entropy_loss(last(preds), 1)
# @time loss, net = backward(net,u,preds,loss,dest) 
# print(loss)
@time training(net, train_data, cross_entropy_loss,1)

# training(net,train_data,cross_entropy_loss)
train_data = load_train_data()
net = intialize_network()
for i in 1:937
    for j in 1:64
        image = train_data[i][1][j]
        label = train_data[i][2][j]
        outputs = forward(net, image)
        loss = cross_entropy_loss(last(outputs), label)
        a = net.conv1.weights
        loss, net = backward(net, image, outputs, loss, label)
        println(output[13])
    end
    
    println(loss, " ", i)
end


function convolution_backwards(input::Array{Float64}, kernel::Array{Float64}, grad::Array{Float64})
    dw = convolution(grad,input)
    pad = size(input,1) - size(grad,1)
    grad_pad = zero_padding(grad, pad)
    dx = convolution(grad_pad, rot180(kernel))
    return dx, dw
end

# @time train_data = load_train_data()

# @time u = train_data[1][1][2]
#@time net = intialize_network()
# o = forward(net, u)
# loss = cross_entropy_loss(last(o), 1)

#     x = fully_connected(x, net.fc2.weights); push!(forward_results,x)
#     x = net.fc2.activation(x); push!(forward_results,x)
# x = convolution(u, net.conv1.weights)
# x = net.conv1.activation(x)  
# x = pooling(x)
# size(u,3)

fc1 = DenseLayer(400, 120, softmax)
fc2 = DenseLayer(120, 84, softmax)
fc3 = DenseLayer(84, 10, softmax)
conv2 = ConvLayer(6, 16, 5, relu)
conv1 = ConvLayer(1, 6, 5, relu)
input = u
length(size(conv1.weights))

x5 = convolution(u,conv1.weights)
y5 = relu(x5)
z5 = pooling(y5)

x4 = convolution(z5,conv2.weights)
y4 = relu(x4)
z4 = pooling(y4)
# # # display_image(z5[:,:,4])
# # # @show z4f = collect(Iterators.flatten(z4))
z4f = vec(z4)

x3 = fully_connected(z4f, fc1.weights)
y3 = relu(x3)

x2 = fully_connected(y3, fc2.weights)
y2 = relu(x2)

x = fully_connected(y2, fc3.weights) 
y = softmax(x)
l = cross_entropy_loss(y, train_data[1][2][2])

# # display_image(input)
# # @show l = cross_entropy_loss(y, train_data[1][2][2])
# # @show y
# # @show label = int_to_array(train_data[1][2][2], 10)
# # label * log.(y)


# # num_of_classes = size(y,1)
# # label = int_to_array(train_data[1][2][2], 10)
# # # y - label' 
# # y - label'
l1 = cross_entropy_loss_backwards(y, train_data[1][2][2])
y1 = softmax_backwards(y) * l1

dx, dw = fully_connected_backwards(y2,fc3.weights,y1)
# # # # y2
dr = relu_backwards(y2) .* dx
dx1, dw1 = fully_connected_backwards(y3,fc2.weights,dr)
# # dw1
dr1 = relu_backwards(y3) .* dx1 
dx2, dw2 = fully_connected_backwards(z4f,fc1.weights,dr1)

re = reshape(dx2,5,5,16)

dm1 = pooling_backwards(y4, z4, re)
# re
# @show dm1[:,:,7]
# @show re
# @show display_image(dm1[:,:,7])
# dm1[:,:,7]
dr3 = relu_backwards(x4) .* dm1
# pad = size(z5,1) - size(dr3,1)
# grad_pad = zero_padding(dr3, pad)
# dr3
#dx = convolution(grad_pad, rot180(kernel))
# dconv2, dcw2 = convolution_backwards(z5,conv2.weights,dr3)
# # dcw2
# convolution(rot180(z5),dr3)

for i in 1:size(z5,3)
    z5[:,:,i] = rot180(z5[:,:,i])
end
convolution(z5,dr3)
for i in 1:size(conv2.weights,3)
    for j in 1:size(conv2.weights,4)
        conv2.weights[:,:,i,j] = rot180(conv2.weights[:,:,i,j])
    end
end
convolution(z5,dr3)
pad = size(z5,1) - size(dr3,1)
grad_pad = zero_padding(dr3, pad)
convolution(grad_pad, conv2.weights)
#conv2.weights
# dm2 = pooling_backwards(y5, z5, dconv2) 
# dr4 = relu_backwards(dm2) .* dm2
# dconv3, dcw3 = convolution_backwards(input,conv1.weights,dr4)
# dcw3
# # y2

using Plots

"""
    display_image(image::Array{Float64,2})
Display a 2D array as an image.

# Arguments
- `image::Array{Float64,2}`: The 2D array representing the image.

"""
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

function pooling(x::Union{Matrix, Vector, Array, Int}, kernel::Int=2, stride::Int=kernel, pool_type::String="max")
    channels = size(x,3)
    output_shape = (size(x,1) - kernel) / stride + 1
    @show output_shape = trunc(Int, output_shape)
    output = zeros((output_shape, output_shape, channels))
    for channel in 1:channels
        for i in 1:stride:size(x,1)
            for j in 1:stride:size(x,2)
                if pool_type == "max"
                    output[fld(i,stride)+1, fld(j,stride)+1, channel] = maximum(x[i:i+kernel-1,j:j+kernel-1])
                elseif pool_type == "mean"
                    output[fld(i,stride)+1, fld(j,stride)+1, channel] = mean(x[i:i+kernel-1,j:j+kernel-1, channel])
                end
            end
        end
    end      
    return output
end
a = [0.9248 0.9596 0.5058 0.1116
    0.1032 0.8585 0.0782 0.4548
    0.8798 0.6298 0.7401 0.1560
    0.7678 0.0892 0.9623 0.4190]
 b = pooling(a,2)
# y4[:,:,1]
# z4[:,:,1]
# re[:,:,1]
c = pooling_backwards(y4, z4, re)

# # a[3:3+2-1,3:3+2-1]


