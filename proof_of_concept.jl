# Layers
struct DenseLayer
    weights::Matrix
    activation::Function
end

struct ConvLayer
    weights::Matrix
    activation::Function
end

function conv_layer(in_channels::Int, out_channels::Int, kernel_size::Int, activation::Function)
    weights = randn(Float32, kernel_size, kernel_size, in_channels, out_channels)
    return ConvLayer(weights, activation)
end

function dense_layer(in_channels::Int, out_channels::Int, activation::Function) 
    weights = randn(Float32, kernel_size, kernel_size, in_channels, out_channels)
    return DenseLayer(weights, activation)


## Functions

# Activation functions
function relu(x::Matrix)
    return max.(0, x)
end


function pooling(x::Matrix, kernel::Int, stride::Int, type::String="max")
    a = size(x)
    output_shape = (size(x,1) - kernel) / stride + 1
    output_shape = trunc(Int, output_shape)
    output = zeros((output_shape, output_shape))

    for i in 1:output_shape
        for j in 1:output_shape
            if type == "max"
                output[i, j] = maximum(x[i:i+stride,j:j+stride])
            elseif type == "mean"
                output[i, j] = mean(x[i:i+stride,j:j+stride])
            end
        end
    end
    
    return output
end

        
function softmax(x::Matrix) 
    exp_x = exp.(x)
    return exp_x ./ sum(exp_x)
end
        
        
function int_to_array(x::Int, length::Int)
    output = zeros(length)
    output[x] = 1
    return output
end
        
# Loss function        
function cross_entropy_loss(pred::Matrix, dest::Int)
    num_of_classes = size(pred,1)
    label = int_to_array(d, num_of_classes)
    loss = -sum(dest .* log.(pred))/N
    return loss
end
        
# Optimizer  
function SGD!(w::Matrix, lr::Number, g::Matrix)
    w .-= lr .* dw
end

# Network
struct ConvNet
    conv1::ConvLayer
    conv2::ConvLayer
    fc1::DenseLayer
    fc2::DenseLayer
    fc3::DenseLayer
end
function net()
    conv1 = conv_layer(1,6,5,relu)
    conv2 = conv_layer(6,16,5,relu)
    fc1 = dense_layer(16*5*5, 120,relu)
    fc2 = dense_layer(120, 84,relu)
    fc3 = dense_layer(84, 10,relu)
    return ConvNet(conv1, conv2, fc1, fc2, fc3)
end

x = [1.0 2.0 3.0 -4.0;
     5.0 -6.0 7.0 8.0;
     9.0 -10.0 -11.0 12.0;
     -13.0 14.0 15.0 16.0]
kernel = 2
stride = 1
out = net


