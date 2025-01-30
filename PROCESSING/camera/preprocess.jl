using Images, Statistics, StatsBase, Base.Threads, ImageEdgeDetection

include("imageprocessing.jl")

imagenames = readdir("testimages")

images = []

cannyalg = Canny(spatial_scale=16, high=ImageEdgeDetection.Percentile(90), low=ImageEdgeDetection.Percentile(60))
#=
imgg = gamma(graynorm(normalize(load("testimages/1_10500_00007.tif"))),g = 1.6)
img1 = histogramadjust(imgg,percentile(vec(imgg), 99.5), .90)
img2 = histogramadjust(img1,percentile(vec(img1), 75), .02)

edges = detect_edges(img2, alg)
mosaicview(img2, edges; nrow=1)
=#

imagesgray = []
imageshist = []
imageshist2 = []
plots = Vector{Matrix{Gray{Float64}}}(undef,length(imagenames))
plots2 = similar(plots)
#edges = []
Threads.@threads for i = 1:1:length(imagenames)
    targetsize = (672,672)#EVEN INTEGERS ONLY!!
    tempimg = normalize(load("testimages/"*imagenames[i]))
    imgsize = size(tempimg)
    #push!(imagesgray,gamma(graynorm(images[i]),g = 1.6))
    tempimg = gamma(graynorm(tempimg),g = 1.6)
    tempimg = histogramadjust(tempimg,percentile(vec(tempimg), 99.5), .90)#TODO, apply these two steps the same way for each propeller on a given day.
    tempimg = histogramadjust(tempimg,percentile(vec(tempimg), 75), .02)
    #medianfilter!(tempimg,5,1) #very slow
    #println(i)
    #edges = detect_edges(tempimg, alg)
    plots[i] = tempimg#push!(plots,tempimg)#mosaicview(tempimg, edges; nrow=1))
    factor = 10
    smallimg = imresize(tempimg, ratio = 1/factor)
    percentilethreshold!(smallimg,75)
    smallimg = erode(smallimg,2)
    cx, cy = round.(whitecentroid(smallimg) .* factor) #TODO: this cropping part should be a function
    xbounds = Int64.((1 + cx - targetsize[1]/2,cx + targetsize[1]/2))
    ybounds = Int64.((1 + cy - targetsize[2]/2,cy + targetsize[2]/2))
    if xbounds[1] < 1
        xbounds = (1,targetsize[1])
    elseif xbounds[2] > imgsize[1]
        xbounds = (1+imgsize[1] - targetsize[1],imgsize[1])
    end
    if ybounds[1] < 1
        ybounds = (1,targetsize[2])
    elseif ybounds[2] > imgsize[2]
        ybounds = (1+imgsize[2] - targetsize[2],imgsize[2])
    end
    tempimg = tempimg[xbounds[1]:xbounds[2],ybounds[1]:ybounds[2]]
    plots2[i] = tempimg
end

#=
ntemp = length(plots[10:end])
sumimg = plots[10] - plots[10]
sumdifimg = plots[10] - plots[10]
meanplots = mean(plots[10:end])

for i = 10:1:ntemp
    global sumdifimg = sumdifimg + abs.(plots[i]-meanplots)#.^2
    global sumimg = sumimg + plots[i]
end
output = medianfilter!(sumdifimg ./ ntemp,25,1)
output = output * (1 / percentile(vec(output), 99.9).val)
output2 = medianfilter!(sumimg ./ ntemp,25,1)
output2 = output2 * (1 / percentile(vec(output2), 99.9).val)

edges =  detect_edges(output .* output2, cannyalg)
edges1 = detect_edges(output, cannyalg)
edges2 = detect_edges(output2, cannyalg)
edges3 = Gray.(erode(dilate(edges,6) .* dilate(edges1,6) .* dilate(edges2,6),4))
=#