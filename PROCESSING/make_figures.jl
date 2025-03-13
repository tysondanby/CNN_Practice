#-----ENVIRONMENT SETUP-------------------------------------------------------
using Pkg
Pkg.activate(@__DIR__) 
required_packages = ["Plots", "CSV", "DataFrames"]
function is_installed(pkg)
    return haskey(Pkg.project().dependencies, pkg)
end
function ensure_packages(packages)
    for pkg in packages
        if !is_installed(pkg)
            println("Installing missing package: ", pkg)
            Pkg.add(pkg)
        end
    end
end
ensure_packages(required_packages)
Pkg.instantiate()
println(@__DIR__)
working_dir = dirname(@__DIR__)
cd(working_dir)
#-----END ENVIRONMENT SETUP---------------------------------------------------

using Plots, CSV, DataFrames
include(pwd()*"/LIB/arrayoperations.jl")
include(pwd()*"/LIB/math.jl")
#include(pwd()*"/LIB/imageprocessing.jl")
threebladed = [4 8]
invert = [1 5]
imagedirectory = "DATA/camera/prepared"
manual_resultscsv = CSV.File("DATA/camera/model_data/manual_results.csv")

function getdatafromimagename(imagename)
    index = findfirst(i->(i==imagename), manual_resultscsv.filename)
    return manual_resultscsv.data[index]
end

function getpointsfromstring(stringofpoints)
    xindex = findfirst(i->(i=='x'), stringofpoints)
    yindex = findfirst(i->(i=='y'), stringofpoints)
    xindicies = []
    yindicies = []
    while xindex != nothing
        push!(xindicies,xindex)
        yindex = findfirst(i->(i=='y'), stringofpoints[(xindex+1):end]) + xindex
        push!(yindicies,yindex)
        step = findfirst(i->(i=='x'), stringofpoints[(xindex+1):end])
        if step != nothing
            xindex = step + xindex
        else
            xindex = nothing
        end
    end
    points = zeros(2,length(xindicies))
    for i = 1:1:(length(xindicies)-1)
        points[1,i] = parse(Float64,stringofpoints[(xindicies[i]+1):(yindicies[i]-1)])
        points[2,i] = parse(Float64,stringofpoints[(yindicies[i]+1):(xindicies[i+1]-1)])
    end
    points[1,end] = parse(Float64,stringofpoints[(xindicies[end]+1):(yindicies[end]-1)])
    points[2,end] = parse(Float64,stringofpoints[(yindicies[end]+1):end])
    return points
end

function makeradial(pointset,propnumber; upsample = 0)
    npoints = size(pointset)[2]
    radialpointset = zeros(2,(npoints-1)*(upsample+1))
    r = distance(pointset[1,1],pointset[2,1],pointset[1,2],pointset[2,2])
    thetabase = atan(pointset[1,2]-pointset[1,1],pointset[2,2]-pointset[2,1])
    for i = 2:1:npoints
        xciel = pointset[1,i]
        yciel = pointset[2,i]
        xprev = pointset[1,i-1]
        yprev = pointset[2,i-1]
        for j = 1:1:(upsample+1)
            frac = j/(upsample+1)
            x = xprev + frac*(xciel-xprev)
            y = yprev + frac*(yciel-yprev)
            index = (i-2)*(upsample+1) + j
            radialpointset[1,index] = distance(pointset[1,1],pointset[2,1],x,y)/r
            radialpointset[2,index] = -1*boundanglepiminuspi(atan(x-pointset[1,1],y-pointset[2,1]) - thetabase)
            
            if contains(invert, propnumber)
                radialpointset[2,index] = -radialpointset[2,index]
            end
            
        end
    end
    return radialpointset
end

function filteroverlapandtip(pointset,rcuttoff)
    npoints = size(pointset)[2]
    newrs = Float64[]
    newths = Float64[]
    tipcavitationflag = false
    rmax = Float64(0.0)
    index = npoints
    endindex = findfirst(x->(x>=0.99),pointset[1,:])
    while (!tipcavitationflag && (index > endindex))#TODO: index > 5 worked if you get a bug here
        r = Float64(pointset[1,index])
        th = Float64(pointset[2,index])
        if (npoints-index)>1
            rmax = maximum(newrs)
        else
            rmax = Float64(0.0)
        end
        if (r < rmax) && ((npoints-index)>2)
            hi = findfirst(x->(x>r),newrs)
            lo = hi-1
            newrs = newrs[1:hi]
            newths = newths[1:hi]
            if (r>rcuttoff) && (findfirst(x->(x>th),newths) == nothing) && false #TODO: false for debug
                tipcavitationflag = true
                push!(newrs,newrs[end])
                push!(newths,2*pi)
                push!(newrs,1.0)
                push!(newths,2*pi)
            else
                newths[end] = (((newths[end]-newths[lo])/(newrs[end]-newrs[lo]))*(r-newrs[lo])) + newths[lo]
                newrs[end] = r
                push!(newrs,r)
                push!(newths,th)
            end
        else
            push!(newrs,r)
            push!(newths,th)
        end
        index = index - 1
    end
    newpointset = zeros(2,length(newrs))
    newpointset[1,:] = newrs
    newpointset[2,:] = newths
    return newpointset
end

function cleanuppointset(pointset;n = 250, rmin = 0.32, rcuttoff = 0.67)#TODO: rcuttoff may be too low
    filteredpointset = filteroverlapandtip(pointset,rcuttoff)
    rclean = collect(range(rmin,1.0; length = n))
    thclean = similar(rclean)
    for (i,r) in enumerate(rclean)
        upperindex = findfirst(x->(x>r), filteredpointset[1,:])
        rhigh = filteredpointset[1,end]
        thhigh = filteredpointset[2,end]
        rlow = 0.0
        thlow = 0.0
        if upperindex != nothing
            rhigh = filteredpointset[1,upperindex]
            thhigh = filteredpointset[2,upperindex]
            if (upperindex - 1) > 0
                rlow = filteredpointset[1,upperindex-1]
                thlow = filteredpointset[2,upperindex-1]
            end
        end
        thclean[i] = (((r-rlow)/(rhigh-rlow)) * (thhigh - thlow)) + thlow
    end
    return rclean,thclean#TODO: splits an array into two vectors. keep this in mind when calling cleanuppointset()
end

function getpointsfromcsv(imagename,propnumber,npointsout)
    datastring = getdatafromimagename(imagename)
    pointsets = []
    nblades = count(i->(i=='B'), datastring)
    startindex = Int64.(zeros(nblades+1))
    startindex[1] = 1
    startindex[end] = length(datastring) + 1
    for i = 2:1:nblades
        startindex[i] = findfirst(i->(i=='B'), datastring[(startindex[i-1]+1):end]) + startindex[i-1]
    end
    thclean = zeros(nblades,npointsout)
    rclean = zeros(npointsout)
    for i = 1:1:nblades
        pointset = getpointsfromstring(datastring[ (startindex[i]+1):(startindex[i+1]-1) ])
        radialpointset = makeradial(pointset,propnumber; upsample = 5)
        rclean,thclean[i,:] = cleanuppointset(radialpointset; n = npointsout)
        push!(pointsets,radialpointset)
    end
    return rclean, thclean
end

#TODO: verify speed better in this function
function getpropspeeddatafromcsv(propnumber,speed;npoints = 250, percentilethreshold = 0.8)
    aliases = [5000 5500;
               5040 5520]
    speedname = deepcopy(speed)
    if propnumber == 5 #Just correcting a stupid mistake I made with file naming.
        if speed == 5000
            speedname = 5040
        elseif speed == 5500
            speedname = 5520
        end
    end
    imagenumbers = [1,11,21,31,41,51,61,71,81,91] #TODO: If I add more samples, this must change.
    rclean = zeros(npoints)
    nblades = 2
    if contains(threebladed, propnumber)
        nblades = 3
    end
    thclean = zeros(nblades*length(imagenumbers),npoints)
    for (i,imagenumber) in enumerate(imagenumbers)
        imagename = "DATASET1_$propnumber"*"_$speedname"*"_"*lpad(imagenumber,5,"0")*".png"
        rclean, thclean[(nblades*(i-1)+1):(nblades*i),:] = getpointsfromcsv(imagename,propnumber,npoints)
    end
    thout = similar(rclean)
    stdev = similar(rclean)
    for (i,r) in enumerate(rclean)
        thout[i], stdev[i] = meanoverpercentile(thclean[:,i],percentilethreshold)
    end
    return rclean, thout .* rclean, stdev .* rclean
end

function plotwitherror(xpts,ypts,stdevpts)
    linecolor = (1.0,0.0,0.0)
    tipvortexcolor = RGBA(0.6,0.7,1.0,1.0)
    yptshigh = ypts + stdevpts
    yptslow = ypts - stdevpts
    shapex = vcat(xpts,reverse(xpts))
    shapey = vcat(yptshigh,reverse(yptslow))
    xbounds = (0,1)
    ybounds = (-0.05,1)
    p = plot(Shape(shapex,shapey);color = RGBA(linecolor... ,0.2),linecolor = RGBA(0,0,0,0), xlims = xbounds,ylims = ybounds, label = "")
    plot!(xpts,ypts, label = "Cavitation Bounds", color = RGBA(linecolor... ,1.0))
    plot!(Shape([xbounds[1],0.32,0.32,xbounds[1]],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = :gray, label = "Propeller Hub")
    index1 = findfirst(x->(x> 1.0),ypts)
    if index1 == nothing
        rtip = 1.1
    else
        rtip = xpts[index1]
    end
    if rtip < 1.0
        plot!(Shape([rtip,xbounds[2],xbounds[2],rtip],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = tipvortexcolor, label = "Tip Vortex Cavitation")
    end
    return p
end

function plotwitherrorboth(xpts,ypts,stdevpts,ypts2,stdevpts2)#TODO: edit this to have titles and legend in the correct spot
    linecolor = (1.0,0.0,0.0)
    linecolor2 = (0.0,1.0,0.0)
    tipvortexcolor = RGBA(0.6,0.7,1.0,1.0)
    tipvortexcolor2 = RGBA(0.6,0.7,1.0,1.0)
    yptshigh = ypts + stdevpts
    yptslow = ypts - stdevpts
    shapex = vcat(xpts,reverse(xpts))
    shapey = vcat(yptshigh,reverse(yptslow))
    yptshigh2 = ypts2 + stdevpts2
    yptslow2 = ypts2 - stdevpts2
    shapex2 = vcat(xpts,reverse(xpts))
    shapey2 = vcat(yptshigh2,reverse(yptslow2))
    xbounds = (0,1)
    ybounds = (-0.05,1)
    
    index1 = findfirst(x->(x> 1.0),ypts)
    index2 = findfirst(x->(x> 1.0),ypts2)
    if index1 == nothing
        rtip = 1.1
    else
        rtip = xpts[index1]
    end
    if index2 == nothing
        rtip2 = 1.1
    else
        rtip2 = xpts[index2]
    end
    if rtip < rtip2
        p = plot(Shape(shapex,shapey);color = RGBA(linecolor... ,0.2),linecolor = RGBA(0,0,0,0), xlims = xbounds,ylims = ybounds, label = "", legend = :topleft)
        plot!(xpts,ypts, label = "Cavitation Bounds - SH", color = RGBA(linecolor... ,1.0))
        if rtip < 1.0
            #TODO plot!(Shape([rtip,xbounds[2],xbounds[2],rtip],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = tipvortexcolor, label = "Tip Vortex Cavitation - SH")
        end
        
        plot!(Shape(shapex2,shapey2);color = RGBA(linecolor2... ,0.2),linecolor = RGBA(0,0,0,0), label = "")
        plot!(xpts,ypts2, label = "Cavitation Bounds - Smooth Hydrophillic", color = RGBA(linecolor2... ,1.0))
        if rtip2 < 1.0
            #TODO plot!(Shape([rtip2,xbounds[2],xbounds[2],rtip2],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = tipvortexcolor2, label = "Tip Vortex Cavitation - Smooth Hydrophillic")
        end
        plot!(Shape([xbounds[1],0.32,0.32,xbounds[1]],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = :gray, label = "Propeller Hub")
        return p
    else
        p = plot(Shape(shapex2,shapey2);color = RGBA(linecolor2... ,0.2),linecolor = RGBA(0,0,0,0), xlims = xbounds,ylims = ybounds, label = "", legend = :topleft)
        plot!(xpts,ypts2, label = "Cavitation Bounds - Smooth Hydrophillic", color = RGBA(linecolor2... ,1.0))
        if rtip2 < 1.0
            #TODO plot!(Shape([rtip2,xbounds[2],xbounds[2],rtip2],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = tipvortexcolor2, label = "Tip Vortex Cavitation - Smooth Hydrophillic")
        end

        plot!(Shape(shapex,shapey);color = RGBA(linecolor... ,0.2),linecolor = RGBA(0,0,0,0), label = "")
        plot!(xpts,ypts, label = "Cavitation Bounds - SH", color = RGBA(linecolor... ,1.0))
        if rtip < 1.0
           #TODO plot!(Shape([rtip,xbounds[2],xbounds[2],rtip],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = tipvortexcolor, label = "Tip Vortex Cavitation - SH")
        end
        plot!(Shape([xbounds[1],0.32,0.32,xbounds[1]],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = :gray, label = "Propeller Hub")
        return p
    end
end

function plotspeed(propnumber,speed)#Deprecated
    rclean, thout, stdev = getpropspeeddatafromcsv(propnumber,speed)
    rclean, thcontrol, stdev2 = getpropspeeddatafromcsv(propnumber,4500)
    return plotwitherror(rclean, thout .- thcontrol, @. sqrt(stdev^2 + stdev2^2))
end

function plotspeedboth(propnumber,speed)#TODO: find the control a different way
    rclean, thout, stdev = getpropspeeddatafromcsv(propnumber,speed)
    rclean, thout2, stdev2 = getpropspeeddatafromcsv(propnumber+4,speed)
    return plotwitherrorboth(rclean, thout, stdev,thout2, stdev2)
end

function createvisualfigures()
    speeds = collect(4500:500:15000)
    props = collect(1:1:4)
    for prop in props
        for speed in speeds
            p = plotspeedboth(prop,speed)
            savefig(p,"DATA/output/figures/highspeed_$prop"*"_$speed"*".png")
        end
    end
end