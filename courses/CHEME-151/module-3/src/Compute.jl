"""
    softmax(x::Array{Float64,1})::Array{Float64,1}

Compute the softmax of a vector. 
This function takes a vector of real numbers and returns a vector of the same size with the softmax of the input vector, 
i.e., the exponential of the input vector divided by the sum of the exponential of the input vector.


### Arguments
- `x::Array{Float64,1}`: a vector of real numbers.

### Returns
- `::Array{Float64,1}`: a vector of the same size as the input vector with the softmax of the input vector.
"""
function softmax(x::Array{Float64,1})::Array{Float64,1}
    
    # compute the exponential of the vector
    y = exp.(x);
    
    # compute the sum of the exponential
    s = sum(y);
    
    # compute the softmax
    return y / s;
end