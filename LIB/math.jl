function circlematrix(r;scalingfactor = 1.0)
    matrix = zeros(2*r+1, 2*r+1)
    for i = 1:1:(2*r+1)
        for j = 1:1:(2*r+1)
            x = (i - r - 1)#*r/(r+.5)
            y = (j - r - 1)#*r/(r+.5)
            if sqrt(x^2+y^2) <= scalingfactor*r
                matrix[i,j] = 1
            end
        end
    end
    return matrix
end

function filterfunc(x ; cutoff = 0.5, factor = 1.0)
    if x <= cutoff
        return cutoff * ((x / cutoff)^factor)
    else
        return 1-((1-cutoff )*(((1-x)/(1-cutoff ))^factor))
    end
end

function pointhistogramadjust(x , initial, final)
    if x < initial
        return x*final/initial
    else
        return (x-initial)*(1-final)/(1-initial) + final
    end
end

function pointscenter(points)
    sum = points[1] .- points[1]
    for i = 1:1:length(points)
        sum = sum .+ points[i]
    end
    return sum ./ length(points)
end

function distance(x1,y1,x2,y2)
    return sqrt((x2-x1)^2 + (y2-y1)^2)
end