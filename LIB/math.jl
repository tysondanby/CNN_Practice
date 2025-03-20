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

function boundanglepiminuspi(theta)
    angle = deepcopy(theta)
    while angle > pi
        angle = angle - 2*pi
    end
    while angle < -pi
        angle = angle + 2*pi
    end
    return angle
end

function meanvec(x)
    return sum(x)/length(x)
end

function stdevvec(x)
    meanval = meanvec(x)
    return sqrt(sum((x.-meanval).^2)/(length(x)-1))
end

function getpercentile(datavector,percentilethreshold)
    nelements = length(datavector)
    increasingvec = sort(datavector)
    lowindex = Int64(floor(percentilethreshold*nelements))
    if (lowindex > 0)
        if (lowindex < nelements)
            lowval = increasingvec[lowindex]
            highval = increasingvec[lowindex+1]
            return (percentilethreshold*nelements - lowindex)*(highval-lowval) + lowval 
        else
            return increasingvec[end]
        end
    else
        return increasingvec[1] #TODO: not technically correct, should be even lower, but this would result in empty sets.
    end
end

function thresholdpercentile(datavector,percentilethreshold)
    newvec = []
    val = getpercentile(datavector,percentilethreshold)
    for element in datavector
        if element >= val
            push!(newvec,element)
        end
    end
    return newvec
end

function meanoverpercentile(datavector,percentilethreshold)
    thresholddata = thresholdpercentile(datavector,percentilethreshold)
    return meanvec(thresholddata), stdevvec(thresholddata)
end

function simulatebrushless(t;freq = 50.0)
    ref = cosd(60.0)
    temp = sin(freq*2*pi*t)
    if temp > ref
        return 1.0
    elseif temp < -ref
        return -1.0
    else
        return 0.0
    end
end