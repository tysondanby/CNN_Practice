include(pwd()*"/LIB/math.jl")
function threshold(img,val)
    imgsize = size(img)
    newimg = zeros(imgsize)
    for row = 1:1:imgsize[1]
        for column = 1:1:imgsize[2]
            if img[row,column] > val
                newimg[row,column] = 1
            end
        end
    end
    return newimg
end

function zeropad(img)
    newimg = zeros(size(img)[1]+2,size(img)[2]+2)
    newimg[2:end-1,2:end-1] = img
    return newimg
end

function onepad(img)
    newimg = ones(size(img)[1]+2,size(img)[2]+2)
    newimg[2:end-1,2:end-1] = img
    return newimg
end

function extend(img)
    newimg = zeros(size(img)[1]+2,size(img)[2]+2)
    newimg[2:end-1,2:end-1] = img
    newimg[1,2:end-1] = img[1,:]
    newimg[end,2:end-1] = img[end,:]
    newimg[2:end-1,1] = img[:,1]
    newimg[2:end-1,end] = img[:,end]
    newimg[1,1] = img[1,1]
    newimg[1,end] = img[1,end]
    newimg[end,1] = img[end,1]
    newimg[end,end] = img[end,end]
    return newimg
end

function filter(img,mask)
    newimg = zeros(size(img))
    workimg = zeros(size(img))
    workimg = img
    rows, columns = size(img)
    masksize = size(mask)[1]
    if (masksize != size(mask)[2]) || (trunc(masksize/2) == round(masksize/2))
        @error("Only square, odd numbered masks are supported.")
    end
    zeropadding = (masksize - 1)/2
    if zeropadding > 0
        for i = 1:1:zeropadding
            workimg = extend(workimg)
        end
    end
    for row = 1:1:rows
        for column = 1:1:columns
            newimg[row,column] = sum(workimg[row:row+masksize-1,column:column+masksize-1] .* mask)
        end
    end
    return newimg
end

function edgefind(img)
    mask = [0  1 0;
            1 -4 1;
            0  1 0]
    return filter(img,mask)
end

function sharpen(img,strength)
    laplacian = edgefind(img)
    return img - strength*laplacian
end

function erode(img,r)
    refimg = deepcopy(img)
    structured::Array{Bool} = circlematrix(r)
    test::Array{Bool} = ones(2*r+1,2*r+1)
    newimg = zeros(size(img))
    onepadding = r
    if onepadding > 0
        for i = 1:1:onepadding
            refimg=onepad(refimg)
        end
    end
    rows,columns = size(img)
    for row = 1:1:rows
        for column = 1:1:columns
            booleanmatrix = (@. Bool(refimg[row:row+(2*r),column:column+(2*r)])) .|| ( .!(structured) )
            if booleanmatrix == test
                newimg[row,column] = 1
            end
        end
    end
    return newimg
end

function dilate(img,r)
    refimg = deepcopy(img)
    structured::Array{Bool} = circlematrix(r)
    test::Array{Bool} = zeros(2*r+1,2*r+1)
    newimg = zeros(size(img))
    zeropadding = r
    if zeropadding > 0
        for i = 1:1:zeropadding
            refimg=zeropad(refimg)
        end
    end
    rows,columns = size(img)
    for row = 1:1:rows
        for column = 1:1:columns
            booleanmatrix = (@. Bool(refimg[row:row+(2*r),column:column+(2*r)])) .&& ( .!(structured) )
            if booleanmatrix != test
                newimg[row,column] = 1
            end
        end
    end
    return newimg
end

function whitecalibrationweights(calibrationimgs,calibrationmasks,targetbrightness)
    @assert length(calibrationimgs) == length(calibrationmasks)
    R, C = size(calibrationimgs[1])
    weights = ones(R,C)
    mastermask = ones(R, C)
    #find master mask
    for i = 1:1:length(calibrationimgs)
        currentmask = calibrationmasks[i]
        for r = 1:1:R
            for c = 1:1:C
                if currentmask[r,c] == 0
                    mastermask[r,c] = 0
                end
            end
        end
    end
    for r = 1:1:R
        for c = 1:1:C
            if mastermask[r,c] == 1
                sum = 0
                for i = 1:1:length(calibrationimgs)
                    currentimg = calibrationimgs[i]
                    sum = sum + currentimg[r,c]
                end
                average = sum/length(calibrationimgs)
                weights[r,c] = deepcopy(targetbrightness/average)
            end
        end
    end

    return weights
end

function contrastadjust!(img,factor)
    R,C = size(img)
    for  r = 1:1:R
        for c = 1:1:C
            oldval = img[r,c]
            newval = oldval + factor * (oldval - 0.5)
            if newval > 1
                img[r,c] = 1
            elseif newval < 0
                img[r,c] = 0
            else
                img[r,c] = newval
            end
        end
    end
    return img
end

function medianfilter!(img,filtersize)#TODO: update like other method of this function
    R,C = size(img)
    extensions = (filtersize-1)/2
    refimg = img
    for i = 1:1:extensions
        refimg = extend(refimg)
    end
    
    for  r = 1:1:R
        for c = 1:1:C
            masked = []
            for fr = 1:1:(filtersize)
                for fc = 1:1:(filtersize)
                    push!(masked,refimg[r+fr-1,c+fc-1])
                end
            end
            img[r,c] = median(masked)
        end
    end
    return img
end

function medianfilter!(img,filtersize,weight)
    R,C = size(img)
    extensions = (filtersize-1)/2
    refimg = img
    for i = 1:1:extensions
        refimg = extend(refimg)
    end
    
    for  r = 1:1:R
        for c = 1:1:C
            masked = refimg[r:(r+filtersize-1),c:(c+filtersize-1)]
            img[r,c] = weight*median(masked) + (1-weight)*img[r,c] 
        end
    end
    return img#img[Int64(extensions+1):end-Int64(extensions),Int64(extensions+1):end-Int64(extensions)]
end

function centeringfilter!(img,filtersize)
    R,C = size(img)
    extensions = (filtersize-1)/2
    refimg = img
    for i = 1:1:extensions
        refimg = extend(refimg)
    end
    
    for  r = 1:1:R
        for c = 1:1:C
            masked = []
            for fr = 1:1:(filtersize)
                for fc = 1:1:(filtersize)
                    push!(masked,refimg[r+fr-1,c+fc-1])
                end
            end
            if (img[r,c] == maximum(masked)) || (img[r,c] == minimum(masked))
                img[r,c] = median(masked)
            end
        end
    end
    return img
end

function rmblackobjects(img,bounds)#removes objects outside of bounds
    xboundmin,xboundmax=bounds[1]
    yboundmin,yboundmax=bounds[2]
    R,C = size(img)
    newimg = deepcopy(img)
    for i =1:1:R
        for j = 1:1:C
            if  (i >= yboundmin) && (i <= yboundmax) && (j >= xboundmin) && (j <= xboundmax) 
            else
                newimg[i,j] = 0
            end
        end
    end
    return newimg
end

function blackcentroid(img)
    pixels::Int64 = 0
    weight = (0.0 , 0.0)
    R,C = size(img)
    for x = 1:1:R
        for y = 1:1:C
            if img[x,y] < 0.5
                pixels = pixels +1
                weight = (weight[1] + x, weight[2] + y)
            end
        end
    end
    cx = weight[1]/pixels
    cy = weight[2]/pixels
    return cx,cy
end

function whitecentroid(img)
    pixels::Int64 = 0
    weight = (0.0 , 0.0)
    R,C = size(img)
    for x = 1:1:R
        for y = 1:1:C
            if img[x,y] > 0.5
                pixels = pixels +1
                weight = (weight[1] + x, weight[2] + y)
            end
        end
    end
    cx = weight[1]/pixels
    cy = weight[2]/pixels
    return cx,cy
end

function graynorm(img::Matrix{RGB{T}}) where T<:Union{AbstractFloat, FixedPoint}
    newimg = @. Gray((red(img)^(1/3))*(green(img)^(1/3))*(blue(img)^(1/3)))
end

function normalize(img::Matrix{RGB{T}}) where T<:Union{AbstractFloat, FixedPoint} #make max fixed point 1
    maxcolor=typemax(typeof(img[1,1].r))
    return img ./ maxcolor
end

function gamma(img::Matrix{T1}; g = 1.0) where T1<:Union{RGB{T2}, Gray{T2}} where T2<:Union{AbstractFloat, FixedPoint}
    return img .^ g
end

function histogramadjust(img::Matrix{T1},intensity, newintensity) where T1<:Union{RGB{T2}, Gray{T2}} where T2<:Union{AbstractFloat, FixedPoint}
    return @. pointhistogramadjust(img, intensity, newintensity)
end

function percentilethreshold!(img::Matrix{T1},pctl) where T1<:Union{RGB{T2}, Gray{T2}} where T2<:Union{AbstractFloat, FixedPoint}
    cutoff = percentile(vec(img), pctl)
    for i = 1:1:length(img)
        if img[i] >= cutoff
            img[i] = 1
        else
            img[i] = 0
        end
    end
end

function filterfartherthan!(img,imgfilter,distance)
    sizeimg = size(img)#should be same as imgfilter TODO: @assert
    #build imgfilter
    usedfilter = copy(imgfilter)
    for i = 1:1:sizeimg[1]
        for j = 1:1:sizeimg[2]
            for  k = 1:1:sizeimg[1]
                for  l = 1:1:sizeimg[2]
                    if imgfilter[k,l] == 1
                        if ((i-k)^2 + (j-l)^2) < (distance^2)
                            usedfilter[i,j] = 1
                            @goto outsidetheloop
                        end
                    end
                end
            end
            @label outsidetheloop
        end
    end
    for i = 1:1:sizeimg[1]
        for j = 1:1:sizeimg[2]
            if (img[i,j] == 1) && (usedfilter[i,j] == 1)
                img[i,j] = 1
            else
                img[i,j] = 0
            end
        end
    end
end

function rotate4dimage90!(fourdimensionalimage,answer,n)
    resolution = size(fourdimensionalimage)[1:2]
    referanceimage =  deepcopy(fourdimensionalimage)
    for i = 1:1:resolution[1]
        for j = 1:1:resolution[2]
            fourdimensionalimage[resolution[1]+1-j,i,1,1] = referanceimage[i,j,1,1]
        end
    end
    answer[1] = typeof(answer[1])(673 - answer[2])
    answer[2] = answer[1]
    θ = atand(answer[4],answer[5])/ n
    θ -= 90.0
    upperbound = 180.0
    if n == 3
        upperbound = 120.0
    end
    while !(0<θ<upperbound)
        θ += upperbound
        if θ >=360.0
            θ -= 360.0
        end
    end
    answer[4] = sind(θ*n)
    answer[5] = cosd(θ*n)
    #=
    for i = 1:1:(length(answer)/2)
        indexx = Int32(2*i-1)
        indexy = Int32(2*i)
        if (answer[indexx] >= 1) && (answer[indexy] >= 1)
            refx = deepcopy(answer[indexx])
            refy = deepcopy(answer[indexy])
            answer[indexx] = typeof(answer[1])(673 - refy)
            answer[indexy] = refx
        end
    end =#
end

function flip4dimageX!(fourdimensionalimage,answer)
    resolution = size(fourdimensionalimage)[1:2]
    referanceimage =  deepcopy(fourdimensionalimage)
    for i = 1:1:resolution[1]
        fourdimensionalimage[673-i,:,1,1] = referanceimage[i,:,1,1]
    end
    answer[1] = typeof(answer[1])(673 - answer[1])
    answer[5] = -answer[5]
end

function flip4dimageX!(fourdimensionalimage)
    resolution = size(fourdimensionalimage)[1:2]
    referanceimage =  deepcopy(fourdimensionalimage)
    for i = 1:1:resolution[1]
        fourdimensionalimage[673-i,:,1,1] = referanceimage[i,:,1,1]
    end
end

function flipimageX!(image)
    resolution = size(image)
    referanceimage =  deepcopy(image)
    for i = 1:1:resolution[1]
        image[673-i,:] = referanceimage[i,:]
    end
end