using Random

# Define sigmoid function
sigmoid(x) = 1 / (1 + exp(-x))

# Define model architecture
struct Model
    w1::Matrix{Float64}
    b1::Vector{Float64}
    w2::Matrix{Float64}
    b2::Vector{Float64}
end

# Define forward pass through the model
function forward(model::Model, x::Vector{Float64})
    a1 = model.w1 * x .+ model.b1
    h1 = sigmoid.(a1)
    a2 = model.w2 * h1 .+ model.b2
    y = softmax(a2)
    return a1, h1, a2, y
end

# Define cross-entropy loss function
function cross_entropy_loss(y_pred, y_true)
    -sum(y_true .* log.(y_pred))
end

# Define function to calculate gradients with automatic differentiation using a graph
function backward(model::Model, x::Vector{Float64}, y_true::Vector{Float64})
    # Forward pass
    a1, h1, a2, y_pred = forward(model, x)

    # Compute loss
    loss = cross_entropy_loss(y_pred, y_true)

    # Initialize gradients
    grad_w1 = zeros(size(model.w1))
    grad_b1 = zeros(size(model.b1))
    grad_w2 = zeros(size(model.w2))
    grad_b2 = zeros(size(model.b2))

    # Backward pass
    graph = [a1, h1, a2, y_pred]
    gradients = [nothing, nothing, nothing, y_pred - y_true]

    # Backpropagate through each layer
    for i = 3:-1:1
        grad = gradients[i + 1]
        input = graph[i]

        # Compute gradients of weights and biases
        grad_w = grad * transpose(graph[i - 1])
        grad_b = grad

        # Update gradients
        gradients[i] = transpose(model.w[i]) * grad
        if i == 3
            gradients[i] .= gradients[i] .* softmax_gradient(a2)
        else
            gradients[i] .= gradients[i] .* sigmoid_gradient(input)
        end
        grad_w .= grad_w / size(x, 2)
        grad_b .= mean(grad_b, dims=2)

        # Update model parameters
        model.w[i] -= learning_rate * grad_w
        model.b[i] -= learning_rate * grad_b
    end

    return loss, model
end

# Define sigmoid gradient function
function sigmoid_gradient(x)
    s = sigmoid.(x)
    s .* (1 .- s)
end

# Define softmax function
function softmax(x)
    exp.(x) ./ sum(exp.(x))
end

# Define softmax gradient function
function softmax_gradient(x)
    s = softmax(x)
    s .* (1 .- s)
end

# Generate random training data
Random.seed!(123)
x_train = rand(2, 1000) .* 2 .- 1
y_train = (sum(x_train, dims=1) .> 0) .+ 1

# Initialize model
learning_rate = 0.1
model = Model(randn(4, 2), randn(4), randn(2, 4), randn(2))


using LinearAlgebra

function convolution(input::Matrix{Float64}, kernel, activation=nothing)
    input_height, input_width = size(input)
    if length(size(kernel)) == 4
        kernel_height, kernel_width, in_channels, out_channels = size(kernel)
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        output = zeros(output_height, output_width, out_channels)

        for in_channel in in_channels
            for out_channel in out_channels
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
    
                for out_channel in out_channels
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

        for input_channel in input_channels
            for in_channel in in_channels
                for out_channel in out_channels
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

        for input_channel in input_channels
            
                for out_channel in out_channels
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
    dw = convolution(input,grad)
    pad = size(input,1) - size(grad,1)
    grad_pad = zero_padding(grad, pad)
    dx = convolution(grad_pad, kernel)
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
        for i in 1:output_shape
            for j in 1:output_shape
                if pool_type == "max"
                    output[i, j, channel] = maximum(x[i:i+stride,j:j+stride, channel])
                elseif pool_type == "mean"
                    output[i, j, channel] = mean(x[i:i+stride,j:j+stride, channel])
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
     for channel in 1:channels
        for (i, local_max) in enumerate(output_flat)
            if i < 10
                for j in i:i+1
                    for k in i:i+1
                        if x[j, k, channel] == local_max
                            pool_grad[j, k, channel] = grad[i]
                        end
                    end
                end
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
    loss = -sum((label .* log.(pred)))/num_of_classes
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
    weights = -0.2 .+ (0.2 - (-0.2)) .* randn(Float32, out_channels, in_channels)
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
    weights = -0.2 .+ (0.2 - (-0.2)) .* randn(Float64, kernel_size, kernel_size, in_channels, out_channels)
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
#train_data = load_train_data()
# train_data[1]

# train_data = load_train_data()

# u = train_data[1][1][1]
# net = intialize_network()
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
input3 = randn(400)
input = u
input2 = randn(120)
x5 = convolution(input,conv1.weights)
y5 = relu(x5)
z5 = pooling(y5)
x4 = convolution(z5,conv2.weights)
y4 = relu(x4)
z4 = pooling(y4) 
z4f = collect(Iterators.flatten(z4))
x3 = fully_connected(z4f, fc1.weights)
y3 = relu(x3)
x2 = fully_connected(y3, fc2.weights)
y2 = relu(x2)
x = fully_connected(y2, fc3.weights) 
y = softmax(x)
l = cross_entropy_loss(y, 1)
num_of_classes = size(y,1)
label = int_to_array(1, num_of_classes)
# y - label' 
l1 = cross_entropy_loss_backwards(y, 1)
y1 = softmax_backwards(y) * l1

dx, dw = fully_connected_backwards(y2,fc3.weights,y1)
y2
dr = relu_backwards(y2) .* dx
dx1, dw1 = fully_connected_backwards(y3,fc2.weights,dr)
dr1 = relu_backwards(y3) .* dx1 
dx2, dw2 = fully_connected_backwards(z4f,fc1.weights,dr1)
dr2 = relu_backwards(dx2) .* dx2
re = reshape(dr2,5,5,16)
dm1 = pooling_backwards(y4, z4, re) 
dr3 = relu_backwards(dm1) .* dm1
dconv2, dcw2 = convolution_backwards(z5,conv2.weights,dr3)
dm2 = pooling_backwards(y5, z5, dconv2) 
dr4 = relu_backwards(dm2) .* dm2
length(size(conv1.weights))
dconv3, dcw3 = convolution_backwards(input,conv1.weights,dr4)
dcw3


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

function backward(model::LeNet5, preds::Vector, loss::Number, dest::Int64, learning_rate::Float64=0.016)
    
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
            println(sequence[i])
        # grad = gradients[i + 1]
        input = graph[i]
        
        if sequence[i] == softmax
            curr_grad = softmax_backwards(curr_grad) * curr_grad
        elseif sequence[i] == relu
            curr_grad = relu_backwards(graph[i-1]) .* curr_grad
        elseif sequence[i] == pooling
            curr_grad = pooling_backwards(graph[i-1],graph[i],curr_grad)
        elseif sequence[i] == "fc3"
            curr_grad, grad_fc3 = fully_connected_backwards(input,model.fc3.weights,curr_grad)
        elseif sequence[i] == "fc2"
            curr_grad, grad_fc2 = fully_connected_backwards(input,model.fc2.weights,curr_grad)
        elseif sequence[i] == "fc1"
            curr_grad, grad_fc1 = fully_connected_backwards(input,model.fc1.weights,curr_grad)
        elseif sequence[i] == "conv2"
            curr_grad, grad_c2 = convolution_backwards(input,conv2.weights,curr_grad)
        elseif sequence[i] == "conv1"
            println(size(input), size(conv1.weights), size(curr_grad))
            curr_grad, grad_c1 = convolution_backwards(input,conv1.weights,curr_grad)
        elseif sequence[i] == "deflatten"
            curr_grad = reshape(curr_grad,5,5,16)
        end
    end
    
    # Update model parameters
    print(size(grad_c2), size(model.conv2.weights))
    model.conv1.weights -= learning_rate .* grad_c1
    model.conv2.weights -= learning_rate .* grad_c2
    model.fc1.weights -= learning_rate .* grad_fc1
    model.fc2.weights -= learning_rate .* grad_fc2
    model.fc3.weights -= learning_rate .* grad_fc3

    return loss, model
end

function training(model::LeNet5, train_data::Vector, criterion::Function, optimizer::Function, num_of_epochs::Int64=10, batch_size::Int64=64)
    num_of_steps = length(train_loader)
    for epoch in 1:num_of_epochs
        for batch in 1:num_of_steps
            images = train_data[batch][1]
            labels = train_data[batch][2]
            loss = []
            for(image,label) in zip(images,labels)
                outputs = forward(model, image)
                push!(loss, criterion(last(outputs), label))
            end
        end
       
    end


train_data = load_train_data()
  
u = train_data[1][1][1]
dest = train_data[1][2][1]
net = intialize_network()
preds = forward(net, u)
loss = cross_entropy_loss(last(preds), 1)
loss, model = backward(net,preds,loss,dest) 


