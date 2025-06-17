using Images, Statistics, StatsBase, Base.Threads

include(pwd()*"/LIB/imageprocessing.jl")

function croparoundpropeller_new(img,targetsize;factor = 10)#NOT WORKING, dont use.
    imgsize = size(img)
    smallimg = imresize(img, ratio = 1/factor)
    smallimghigh = copy(smallimg)
    percentilethreshold!(smallimg,75)#was 75
    percentilethreshold!(smallimghigh,99)
    filterfartherthan!(smallimg,smallimghigh,round(150/factor))#filters smallimg such that only highs occur within 100px of the very brightest points
    smallimg = erode(smallimg,2)
    cx, cy = round.(whitecentroid(smallimg) .* factor)
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
    return img[xbounds[1]:xbounds[2],ybounds[1]:ybounds[2]]
end

function croparoundpropeller(img,targetsize;factor = 10)
    imgsize = size(img)
    smallimg = imresize(img, ratio = 1/factor)
    smallimgbackup = copy(smallimg)
    sumsmall = 0.0
    percentile = 99
    while sumsmall < 1.0
        smallimg = copy(smallimgbackup)
        percentilethreshold!(smallimg,percentile)#was 75
        smallimg = erode(smallimg,1)#was 2
        sumsmall=sum(smallimg)
        percentile = percentile - 0.25
    end
    cx, cy = round.(whitecentroid(smallimg) .* factor)
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
    return img[xbounds[1]:xbounds[2],ybounds[1]:ybounds[2]]
end

function preprocessimages()
    imagenames = readdir("DATA/camera/raw")
    nimages = length(imagenames)
    targetsize = (672,672)#EVEN INTEGERS ONLY!!
    progress = 0
    re_lock = ReentrantLock()
    Threads.@threads for i = 1:1:nimages
        rawfilename = "DATA/camera/raw/"*imagenames[i]
        preparedfilename = "DATA/camera/prepared/"*imagenames[i][1:end-4]*".png"
        if (!isfile(preparedfilename)) && (!contains(rawfilename,".csv"))
            using Measures
            tempimg = load(rawfilename)#normalize(load(rawfilename))#
            tempimg = graynorm(tempimg)
            #Images.save("step1.png",tempimg)
            #h = fit(Histogram,Float64.(vec(tempimg)); nbins = 150).weights
            #h = clamp.(h, 1, maximum(h))
            #p = plot(h,yaxis=:log, fillrange=1, fillalpha=0.5, legend = false, ticks = false, color = RGB(0.3,0.3,0.3), xlabel = "Relative Pixel Intensity", ylabel = "Relative Frequency", size=(600, 400), left_margin = 5mm, bottom_margin = 2mm,xlims = (1,length(h)),ylims = (1,maximum(h)))
            #savefig(p,"step1hist.png")
            tempimg = gamma(tempimg,g = 1.6)
            #Images.save("step2.png",tempimg)
            #h = fit(Histogram,Float64.(vec(tempimg)); nbins = 150).weights
            #h = clamp.(h, 1, maximum(h))
            #p = plot(h,yaxis=:log, fillrange=1, fillalpha=0.5, legend = false, ticks = false, color = RGB(0.3,0.3,0.3), xlabel = "Relative Pixel Intensity", ylabel = "Relative Frequency", size=(600, 400), left_margin = 5mm, bottom_margin = 2mm,xlims = (1,length(h)),ylims = (1,maximum(h)))
            #savefig(p,"step2hist.png")
            tempimg = histogramadjust(tempimg,percentile(vec(tempimg), 99.5), .90)#TODO, apply these two steps the same way for each propeller on a given day.
            #Images.save("step3.png",tempimg)
            #h = fit(Histogram,Float64.(vec(tempimg)); nbins = 150).weights
            #h = clamp.(h, 1, maximum(h))
            #p = plot(h,yaxis=:log, fillrange=1, fillalpha=0.5, legend = false, ticks = false, color = RGB(0.3,0.3,0.3), xlabel = "Relative Pixel Intensity", ylabel = "Relative Frequency", size=(600, 400), left_margin = 5mm, bottom_margin = 2mm,xlims = (1,length(h)),ylims = (1,maximum(h)))
            #savefig(p,"step3hist.png")
            tempimg = histogramadjust(tempimg,percentile(vec(tempimg), 75), .02)
            #Images.save("step4.png",tempimg)
            #h = fit(Histogram,Float64.(vec(tempimg)); nbins = 150).weights
            #h = clamp.(h, 1, maximum(h))
            #p = plot(h,yaxis=:log, fillrange=1, fillalpha=0.5, legend = false, ticks = false, color = RGB(0.3,0.3,0.3), xlabel = "Relative Pixel Intensity", ylabel = "Relative Frequency", size=(600, 400), left_margin = 5mm, bottom_margin = 2mm,xlims = (1,length(h)),ylims = (1,maximum(h)))
            #savefig(p,"step4hist.png")
            croppedimg = croparoundpropeller(tempimg,targetsize)
            #Images.save("step5.png",tempimg)
            Images.save(preparedfilename,croppedimg) #TODO: Save
        end
        @lock re_lock progress = progress + 1
        @lock re_lock println("Preprocessed $progress / $nimages images.")
    end
end